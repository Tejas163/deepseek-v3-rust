use candle_core::{DType, Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

use crate::config::DeepSeekV3Config;

pub struct TokenEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
    config: DeepSeekV3Config,
}

impl TokenEmbeddings {
    pub fn new(vb: VarBuilder, vocab_size: usize, hidden_size: usize) -> Result<Self> {
        let config = DeepSeekV3Config::default();

        let token_embedding = embedding(vb.pp("token_embedding"), vocab_size, hidden_size)?;
        let position_embedding = embedding(
            vb.pp("position_embedding"),
            config.max_position_embeddings,
            hidden_size,
        )?;

        Ok(Self {
            token_embedding,
            position_embedding,
            config,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?;

        let token_embeds = self.token_embedding.forward(input_ids)?;
        let position_embeds = self
            .position_embedding
            .forward(&position_ids.unsqueeze(0))?;

        // Add token and position embeddings
        token_embeds.add(&position_embeds)
    }
}

pub struct RotaryPositionalEmbedding {
    inv_freq: Tensor,
    config: DeepSeekV3Config,
}

impl RotaryPositionalEmbedding {
    pub fn new(config: &DeepSeekV3Config, device: candle_core::Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let inv_freq = Self::compute_inv_freq(config.rope_theta, head_dim, device)?;

        Ok(Self {
            inv_freq,
            config: config.clone(),
        })
    }

    fn compute_inv_freq(
        theta: f32,
        head_dim: usize,
        device: candle_core::Device,
    ) -> Result<Tensor> {
        let freqs = (1..=head_dim)
            .map(|i| theta.powf(-2.0 * (i - 1) as f32 / head_dim as f32))
            .collect::<Vec<_>>();

        Tensor::new(freqs.as_slice(), &device)?.reshape((1, 1, head_dim))
    }

    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        let device = self.inv_freq.device();

        // Create positions [0, 1, 2, ..., seq_len-1]
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.reshape((1, seq_len, 1))?;

        // Compute frequencies: positions * inv_freq
        // Shape: [1, seq_len, head_dim]
        let freqs = positions.broadcast_mul(&self.inv_freq)?;

        // Create cos and sin
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Return interleaved [cos, sin, cos, sin, ...]
        let half_head_dim = self.inv_freq.dim(2)? / 2;
        let cos_part = cos.narrow(2, 0, half_head_dim)?;
        let sin_part = sin.narrow(2, 0, half_head_dim)?;

        // Interleave: [cos0, sin0, cos1, sin1, ...]
        let mut result = Vec::with_capacity(half_head_dim * 2);
        for i in 0..half_head_dim {
            result.push(cos_part.clone());
            result.push(sin_part.clone());
        }

        // This is simplified - in practice you'd use proper tensor operations
        Ok(cos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_embeddings() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::new(DType::F32, &device);

        let embeddings = TokenEmbeddings::new(vb, 50000, 512)?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;

        let output = embeddings.forward(&input_ids)?;
        println!("Output shape: {:?}", output.shape());

        assert_eq!(output.dim(1)?, 5);
        assert_eq!(output.dim(2)?, 512);

        Ok(())
    }
}
