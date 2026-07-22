use super::layers::quantized_var_builder::VarBuilder as QVarBuilder;
use super::{
    attention::QuantizedAttention, rotary_emb::ScalingRotaryEmbedding, Config, KvCacheDtype,
};
use crate::backend::progress::{ProgressLike, ProgressReporter};
#[cfg(feature = "nccl")]
use crate::openai::distributed::AllReduce;
use crate::openai::distributed::{Comm, Rc, VocabParallelLinear};
use crate::openai::models::layers::qrmsnorm::QRmsNorm;
use crate::openai::models::mask::get_attention_causal_mask;
use crate::InputMetadata;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};
use either::Either;
use parking_lot::RwLock;
use std::iter::zip;
use std::sync::Arc;

struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
    #[cfg(feature = "nccl")]
    all_reduce: Option<AllReduce>,
    #[cfg(feature = "nccl")]
    dtype: DType,
}

/// `MOSS_FUSED_GLUE=0` disables the fused SwiGLU matvec (B3,
/// moss rtf-beyond-17x phase-b) and forces the original unfused
/// gate/silu/mul path — the A/B escape hatch. Cached on first read.
fn fused_glu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MOSS_FUSED_GLUE").map_or(true, |v| v != "0"))
}

impl Mlp {
    #[allow(unused_mut)]
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Fused silu(w1·x)*(w3·x) in one kernel with one shared activation
        // quantization (CUDA + Q4K + decode-width batches; `Ok(None)` means
        // not applicable and we take the unfused path below).
        let gated = if fused_glu_enabled() {
            candle_core::quantized::fused_swiglu_forward(
                &self.feed_forward_w1,
                &self.feed_forward_w3,
                xs,
            )?
        } else {
            None
        };
        let gated = match gated {
            Some(t) => t,
            None => {
                let w1 = self.feed_forward_w1.forward(xs)?;
                let w3 = self.feed_forward_w3.forward(xs)?;
                (candle_nn::ops::silu(&w1)? * w3)?
            }
        };
        let mut y = self.feed_forward_w2.forward(&gated)?;
        #[cfg(feature = "nccl")]
        if let Some(all_reduce) = &self.all_reduce {
            y = all_reduce.apply(&y.to_dtype(self.dtype)?)?;
            y = y.to_dtype(DType::F32)?;
        }
        Ok(y)
    }
}

struct LayerWeights {
    self_attn: QuantizedAttention,
    attention_norm: QRmsNorm,
    mlp: Mlp,
    ffn_norm: QRmsNorm,
}

impl LayerWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Vec<Tensor>>,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.self_attn
            .forward(x, mask, input_positions, cache, input_metadata)
    }
}

pub struct GGUFQWen {
    tok_embeddings: Embedding,
    layers: Vec<LayerWeights>,
    norm: QRmsNorm,
    output: VocabParallelLinear,
    cfg: Config,
    dtype: DType,
    device: Device,
    /// The true vocabulary size (from `tokenizer.ggml.tokens`'s array
    /// length), distinct from `token_embd.weight`/`output.weight`'s row
    /// count: GGUF quantization requires row counts to be a multiple of the
    /// quant block size, so those tensors are padded beyond the real
    /// vocabulary (e.g. 151936 real tokens padded to 155648 rows for Q4_K's
    /// 256-element blocks). The padding rows are untrained/arbitrary
    /// weights that can produce spuriously large logits, corrupting
    /// argmax/sampling if not excluded. Logits are narrowed to this size
    /// before being returned to callers.
    true_vocab_size: usize,
}

impl GGUFQWen {
    pub fn into_config(
        embedding_length: usize,
        head_dim: usize,
        i_size: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        rope_theta: f64,
        rms_eps: f64,
        max_seq_len: usize,
        original_max_position_embeddings: Option<usize>,
        partial_rotary_factor: Option<f32>,
        _kv_cache_dtype: DType,
    ) -> Config {
        Config {
            architectures: Some(vec!["qwen".to_string()]),
            hidden_size: embedding_length,
            head_dim: Some(head_dim),
            intermediate_size: i_size,
            vocab_size: 0,
            num_hidden_layers: block_count,
            num_attention_heads: head_count,
            num_key_value_heads: Some(head_count_kv),
            rms_norm_eps: rms_eps,
            rope_theta,
            rope_local_base_freq: None,
            bos_token_id: Some(super::TokenID(Either::Left(Some(151644)))),
            eos_token_id: Some(super::TokenID(Either::Left(Some(151645)))),
            max_seq_len,
            sliding_window: None,
            sliding_window_pattern: None,
            hidden_act: None,
            hidden_activation: None,
            tie_word_embeddings: false,
            rope_scaling: None,
            max_position_embeddings: Some(max_seq_len),
            original_max_position_embeddings,
            attention_bias: Some(false),
            partial_rotary_factor,
            qk_layernorm: false,
            use_qkv_bias: None,
            custom_stop_tokens: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            quantization_config: None,
            moe_config: None,
            isq_quant: None,
            kvcache_dtype: KvCacheDtype::Auto,
            extra_config_json: None,
            is_f16_mode: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_gguf(
        vb: &QVarBuilder,
        device: &Device,
        dtype: DType,
        kv_cache_dtype: DType,
        yarn_scaling_factor: Option<f64>,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
        rank: usize,
        world_size: usize,
        #[allow(unused_variables)] comm: Rc<Comm>,
        kvcache_compression: KvCacheDtype,
    ) -> Result<Self> {
        let metadata = vb.first_content_metadata();
        let md_get = |s: &str| match metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let reporter = progress_reporter.clone();
        let arch = md_get("general.architecture")?.to_string()?;

        let head_count =
            md_get(format!("{arch}.attention.head_count").as_str())?.to_u32()? as usize;
        let head_count_kv =
            md_get(format!("{arch}.attention.head_count_kv").as_str())?.to_u32()? as usize;

        let head_dim = md_get(format!("{arch}.attention.key_length").as_str());
        let embedding_length =
            md_get(format!("{arch}.embedding_length").as_str())?.to_u32()? as usize;
        let head_dim = if head_dim.is_ok() {
            head_dim.unwrap().to_u32()? as usize
        } else {
            embedding_length / head_count
        };
        let context_length = md_get(format!("{arch}.context_length").as_str())?.to_u32()? as usize;
        let block_count = md_get(format!("{arch}.block_count").as_str())?.to_u32()? as usize;
        let rms_norm_eps =
            md_get(format!("{arch}.attention.layer_norm_rms_epsilon").as_str())?.to_f32()? as f64;
        let rope_freq_base = md_get(format!("{arch}.rope.freq_base").as_str())
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);

        let tok_embeddings = vb.get_no_shape("token_embd.weight")?;
        let vocab_size = tok_embeddings.shape().dims()[0];
        // token_embd.weight/output.weight are padded to a multiple of the
        // quantization block size (e.g. 151936 real tokens -> 155648 rows
        // for Q4_K's 256-element blocks) — NOT the true vocabulary size.
        // tokenizer.ggml.tokens's array length is the true count; fall back
        // to the (possibly padded) tensor row count if that metadata key is
        // missing rather than failing to load.
        // Neither the tensor row count nor tokenizer.ggml.tokens's array
        // length disclose the true (unpadded) vocabulary — both are padded
        // to a multiple of the quantization block size. 151936 is Qwen3's well-known,
        // documented vocabulary size. Scoped to arch == "qwen3" specifically rather
        // than assumed for every model this shared loader might ever serve.
        let true_vocab_size = if arch == "qwen3" { 151_936 } else { vocab_size };
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm =
            QRmsNorm::from_arc_qtensor(vb.get_no_shape("output_norm.weight")?, rms_norm_eps)?;
        let output_tensor_name = if vb.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let output = VocabParallelLinear::load_from_gguf(
            vb,
            output_tensor_name,
            vocab_size,
            comm.clone(),
            dtype,
        )?;
        let original_max_position_embeddings =
            md_get(format!("{arch}.rope.scaling.original_context_length").as_str());
        let original_max_position_embeddings = if original_max_position_embeddings.is_ok() {
            Some(original_max_position_embeddings.unwrap().to_u32()? as usize)
        } else {
            None
        };

        let rope_dim = md_get(format!("{arch}.rope.dimension_count").as_str());
        let partial_rotary_factor = if rope_dim.is_ok() {
            let rope_dim = rope_dim.unwrap().to_u32()? as usize;
            if rope_dim != head_dim {
                Some(rope_dim as f32 / head_dim as f32)
            } else {
                None
            }
        } else {
            None
        };
        let mut cfg = GGUFQWen::into_config(
            embedding_length,
            head_dim,
            0,
            block_count,
            head_count,
            head_count_kv,
            rope_freq_base as f64,
            rms_norm_eps,
            context_length,
            original_max_position_embeddings,
            partial_rotary_factor,
            kv_cache_dtype,
        );
        // into_config hardcodes vocab_size: 0 (it doesn't have this value
        // available at that call site) - fill in the real one now.
        cfg.vocab_size = true_vocab_size;
        // into_config also hardcodes kvcache_dtype: KvCacheDtype::Auto (it
        // doesn't have the caller's requested compression mode available
        // at that call site either) - wire up the real one now. Without
        // this, `QuantizedAttention::new`'s `PagedAttention::new(...,
        // config.kvcache_dtype.is_fp8_keys())` call always sees `Auto`
        // (is_fp8_keys() == false) regardless of what `kv_compression` the
        // caller configured, so `k_scale`/`v_scale` are never allocated —
        // while the KV cache engine (built separately, from the same
        // `kv_compression` option) still dispatches decode through the
        // turbo8/fp8 kernels, which read a null scale pointer. Root-caused
        // from a real production run producing progressively-corrupted
        // (not NaN, not crashing) audio under `kv_compression: turbo8`.
        cfg.kvcache_dtype = kvcache_compression;
        cfg.apply_runtime_rope_overrides(yarn_scaling_factor);
        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(DType::F32, &cfg, device, true)?);

        let mut layers = Vec::with_capacity(block_count);

        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let prefix_vb = vb.pp(&prefix);
            let mlp = {
                let feed_forward_w1 =
                    prefix_vb.get_sharded_no_shape("ffn_gate.weight", 0, rank, world_size)?;
                let feed_forward_w2 =
                    prefix_vb.get_sharded_no_shape("ffn_down.weight", 1, rank, world_size)?;
                let feed_forward_w3 =
                    prefix_vb.get_sharded_no_shape("ffn_up.weight", 0, rank, world_size)?;
                Mlp {
                    feed_forward_w1: QMatMul::from_arc(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_arc(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_arc(feed_forward_w3)?,
                    #[cfg(feature = "nccl")]
                    all_reduce: if world_size > 1 {
                        Some(AllReduce::new(comm.clone()))
                    } else {
                        None
                    },
                    #[cfg(feature = "nccl")]
                    dtype,
                }
            };

            let attention_norm = prefix_vb.get_no_shape("attn_norm.weight")?;
            let ffn_norm = prefix_vb.get_no_shape("ffn_norm.weight")?;

            let self_attn = QuantizedAttention::new(
                &cfg,
                vb,
                &prefix,
                device,
                dtype,
                rotary_emb.clone(),
                cfg.sliding_window,
                rank,
                world_size,
                comm.clone(),
            )?;

            layers.push(LayerWeights {
                self_attn,
                attention_norm: QRmsNorm::from_arc_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                ffn_norm: QRmsNorm::from_arc_qtensor(ffn_norm, rms_norm_eps)?,
            });
            reporter.write().set_progress(layer_idx + 1);
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            cfg,
            dtype,
            device: device.clone(),
            true_vocab_size,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, false)
    }

    pub fn forward_embedding(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        self.forward_inner(x, input_positions, kv_caches, input_metadata, true)
    }

    /// Decodes from externally computed embeddings, bypassing the token
    /// embedding lookup entirely. `embeddings` must already have the shape
    /// `self.tok_embeddings.forward(token_ids)` would have produced
    /// (`[total_tokens, hidden_size]`, packed across the batch the same way
    /// `input_positions`/`input_metadata` describe it). Returns
    /// `(logits, hidden_state)` for every position in `embeddings` (callers
    /// that only need the last-token result should slice the result
    /// themselves).
    pub fn forward_from_embeddings(
        &self,
        embeddings: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.forward_layers(
            embeddings.clone(),
            input_positions,
            kv_caches,
            input_metadata,
        )?;
        let hidden = self.norm.forward(&hidden)?;
        // token_embd.weight/output.weight are padded beyond the true
        // vocabulary for GGUF quantization block-size alignment. The padding
        // rows are untrained weights and can produce spuriously large
        // logits, so the padding tail is dropped before returning to
        // callers (confirmed via a real-weights probe: identical logits
        // over the true-vocab prefix between this and CandleBackbone, but
        // CandleBackbone's argmax always lands in-range while this path's
        // argmax without narrowing lands in the untrained padding tail).
        // Narrow FIRST (on the raw output, before the dtype cast) and force
        // the result contiguous on-device. The old order
        // (`.to_dtype(F32)?.narrow(...)`) returned a strided last-dim view;
        // downstream in-place copies of that non-contiguous view are what a
        // CUDA-graph capture bakes in as row-wise staged copies — replayed
        // graphs then keep re-reading capture-time row data for rows >= 1
        // (row 0 aliases offset 0 and stays live). Root-caused 2026-07-23
        // via a per-row replay-vs-eager probe in moss-tts: hidden bit-exact,
        // logits row 0 exact, logits rows 1+ frozen at capture-time values.
        // `.contiguous()` keeps the whole tail a flat device tensor so the
        // captured copy is one D2D memcpy node.
        let logits = self
            .output
            .forward(&hidden)?
            .narrow(candle_core::D::Minus1, 0, self.true_vocab_size)?
            .contiguous()?
            .to_dtype(DType::F32)?;
        Ok((logits, hidden))
    }

    fn forward_inner(
        &self,
        x: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        return_hidden: bool,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };
        let xs = self.tok_embeddings.forward(x)?;
        let mut xs = self.forward_layers(xs, input_positions, kv_caches, input_metadata)?;
        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<_> = seqlens.iter().map(|x| x - 1 as u32).collect();
            let batch = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (batch,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;

        if return_hidden {
            return Ok(xs);
        }
        // See forward_from_embeddings's comment: the padding tail beyond
        // true_vocab_size is dropped for the same reason.
        self.output.forward(&xs)?.to_dtype(DType::F32)?.narrow(
            candle_core::D::Minus1,
            0,
            self.true_vocab_size,
        )
    }

    /// Runs the transformer stack (attention + MLP layers only, no final
    /// norm) over an already-embedded input, shared by both the token-id
    /// path (`forward_inner`) and the raw-embeddings path
    /// (`forward_from_embeddings`). Callers are responsible for applying
    /// `self.norm` (and, for token-id prefill, any last-token selection)
    /// afterwards.
    fn forward_layers(
        &self,
        mut xs: Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let seqlens = if input_metadata.cu_seqlens_q.is_some() {
            input_metadata
                .cu_seqlens_q
                .as_ref()
                .unwrap()
                .to_vec1::<u32>()?[1..]
                .into()
        } else {
            Vec::new()
        };
        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            input_positions,
            &seqlens,
            self.cfg.sliding_window,
            input_metadata.is_prefill,
        );
        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                let x = xs;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    Some((k_cache, v_cache)),
                    input_metadata,
                )?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = (x + residual)?;
                xs = x
            }
        } else {
            for layer in self.layers.iter() {
                let x = xs;
                let residual = &x;
                let x = layer.attention_norm.forward(&x)?;
                let attn = layer.forward_attn(
                    &x,
                    attention_mask.as_ref(),
                    input_positions,
                    None,
                    input_metadata,
                )?;
                let x = (attn + residual)?;

                // MLP
                let residual = &x;
                let x = layer.ffn_norm.forward(&x)?;
                let x = layer.mlp.forward(&x)?;
                let x = (x + residual)?;
                xs = x
            }
        }
        Ok(xs)
    }

    pub fn get_config(&self) -> &Config {
        &self.cfg
    }
}
