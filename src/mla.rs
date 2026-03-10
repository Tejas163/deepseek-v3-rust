use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::DeepSeekV3Config;

pub struct MultiHeadLatentAttention {
    q_proj: Linear,
    kv_proj: Linear,  # Combined K-V projection with latent dim
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    latent_dim: usize,  # Compressed KV dimension
    config: DeepSeekV3Config,
}

impl MultiHeadLatentAttention {
    pub fn new(vb: VarBuilder, config: &DeepSeekV3Config) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        // Latent KV dimension (compressed) - typically much smaller than full KV
        let latent_dim = head_dim * config.num_kv_heads / 2;  # Compress to 50%
        
        let q_proj = linear(vb.pp("q_proj"), config.hidden_size, config.hidden_size)?;
        let kv_proj = linear(vb.pp("kv_proj"), config.hidden_size, latent_dim * 2)?;  # Both K and V
        let o_proj = linear(vb.pp("o_proj"), config.hidden_size, config.hidden_size)?;
        
        Ok(Self {
            q_proj,
            kv_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            latent_dim,
            config: config.clone(),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        # Project to Q and compressed KV
        let q = self.q_proj.forward(x)?;  # [batch, seq, hidden]
        let kv = self.kv_proj.forward(x)?; # [batch, seq, latent*2]
        
        # Split KV into K and V (latent dimension)
        let latent_per_kv = self.latent_dim;
        let k_latent = kv.narrow(2, 0, latent_per_kv)?;
        let v_latent = kv.narrow(2, latent_per_kv, latent_per_kv)?;
        
        # Reshape Q for multi-head attention
        # [batch, seq, hidden] -> [batch, seq, num_heads, head_dim] -> [batch, num_heads, seq, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let q = q.transpose(1, 2)?;
        
        # For simplicity, we expand latent K/V to full dimension (in practice, use upcasting)
        let k = k_latent.broadcastreshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let k = k.transpose(1, 2)?;
        let v = v_latent.broadcastreshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?;
        
        # Compute attention scores
        # Q @ K^T / sqrt(head_dim)
        let scale = (self.head_dim as f32).sqrt().recip();
        let k_t = k.transpose(2, 3)?;  # [batch, num_heads, head_dim, seq]
        let attn_scores = q.matmul(&k_t)?.mul(scale)?;
        
        # Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_scores, 3)?;
        
        # Apply attention to V
        let attn_output = attn_weights.matmul(&v)?;  # [batch, num_heads, seq, head_dim]
        
        # Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output.transpose(1, 2)?.reshape((batch_size, seq_len, self.config.hidden_size))?;
        
        # Output projection
        self.o_proj.forward(&attn_output)
    }
}

# Helper function to create attention mask for causal attention
pub fn create_causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    # Create lower triangular mask
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    
    Tensor::new(mask.as_slice(), device)?.reshape((1, 1, seq_len, seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_mla() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::new(DType::F32, &device);
        let config = DeepSeekV3Config::default();
        
        let mla = MultiHeadLatentAttention::new(vb, &config)?;
        let x = Tensor::randn(0f32, 1f32, (2, 8, 512), &device)?;
        
        let output = mla.forward(&x)?;
        println!("MLA Output shape: {:?}", output.shape());
        
        assert_eq!(output.dims3()?, (2, 8, 512));
        
        Ok(())
    }
}
