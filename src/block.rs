use candle_core::{DType, Result, Tensor};
use candle_nn::{rms_norm, RMSNorm, VarBuilder};

use crate::config::DeepSeekV3Config;
use crate::mla::MultiHeadLatentAttention;
use crate::moe::MoELayer;

pub struct DeepSeekBlock {
    attention: MultiHeadLatentAttention,
    moe: MoELayer,
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
    config: DeepSeekV3Config,
}

impl DeepSeekBlock {
    pub fn new(vb: VarBuilder, layer_idx: usize, config: &DeepSeekV3Config) -> Result<Self> {
        let vb = vb.pp(format!("layer_{}", layer_idx));
        
        let attention = MultiHeadLatentAttention::new(vb.pp("attention"), config)?;
        let moe = MoELayer::new(vb.pp("moe"), config)?;
        
        let attn_norm = rms_norm(vb.pp("attn_norm"), config.hidden_size, config.rms_norm_eps)?;
        let ffn_norm = rms_norm(vb.pp("ffn_norm"), config.hidden_size, config.rms_norm_eps)?;
        
        Ok(Self {
            attention,
            moe,
            attn_norm,
            ffn_norm,
            config: config.clone(),
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        # Pre-norm architecture (like DeepSeek V3)
        
        # === Attention Block ===
        let attn_input = self.attn_norm.forward(x)?;
        let attn_output = self.attention.forward(&attn_input)?;
        let x = x.add(&attn_output)?;  # Residual connection
        
        # === MoE Block ===
        let ffn_input = self.ffn_norm.forward(&x)?;
        let (moe_output, aux_loss) = self.moe.forward(&ffn_input)?;
        let x = x.add(&moe_output)?;  # Residual connection
        
        Ok((x, aux_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_block() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::new(DType::F32, &device);
        let config = DeepSeekV3Config::default();
        
        let block = DeepSeekBlock::new(vb, 0, &config)?;
        let x = Tensor::randn(0f32, 1f32, (2, 8, 512), &device)?;
        
        let (output, aux_loss) = block.forward(&x)?;
        println!("Block Output shape: {:?}", output.shape());
        println!("Aux loss: {:?}", aux_loss);
        
        assert_eq!(output.dims3()?, (2, 8, 512));
        
        Ok(())
    }
}
