use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::config::DeepSeekV3Config;
use crate::embeddings::TokenEmbeddings;
use crate::block::DeepSeekBlock;

pub struct DeepSeekV3 {
    embeddings: TokenEmbeddings,
    layers: Vec<DeepSeekBlock>,
    final_norm: candle_nn::RMSNorm,
    lm_head: Linear,
    config: DeepSeekV3Config,
}

impl DeepSeekV3 {
    pub fn new(vb: VarBuilder, config: &DeepSeekV3Config) -> Result<Self> {
        let embeddings = TokenEmbeddings::new(vb.pp("embeddings"), config.vocab_size, config.hidden_size)?;
        
        let layers: Vec<DeepSeekBlock> = (0..config.num_hidden_layers)
            .map(|i| DeepSeekBlock::new(vb.pp("layers"), i, config))
            .collect::<Result<Vec<_>>>()?;
        
        let final_norm = candle_nn::rms_norm(vb.pp("final_norm"), config.hidden_size, config.rms_norm_eps)?;
        
        # Language modeling head (tied with embeddings)
        let lm_head = linear(vb.pp("lm_head"), config.hidden_size, config.vocab_size)?;
        
        Ok(Self {
            embeddings,
            layers,
            final_norm,
            lm_head,
            config: config.clone(),
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        # Get embeddings
        let hidden_states = self.embeddings.forward(input_ids)?;
        
        # Pass through transformer layers
        let mut aux_losses = Vec::new();
        let mut hidden_states = hidden_states;
        
        for layer in &self.layers {
            let (output, aux_loss) = layer.forward(&hidden_states)?;
            hidden_states = output;
            if let Some(loss) = aux_loss {
                aux_losses.push(loss);
            }
        }
        
        # Final normalization
        let hidden_states = self.final_norm.forward(&hidden_states)?;
        
        # LM head
        let logits = self.lm_head.forward(&hidden_states)?;
        
        # Sum auxiliary losses
        let total_aux_loss = if aux_losses.is_empty() {
            None
        } else {
            Some(aux_losses.iter().fold(Tensor::zeros(1, input_ids.device())?, |acc, x| acc.add(x)?)?)
        };
        
        Ok((logits, total_aux_loss))
    }
    
    # Generate text using greedy decoding
    pub fn generate(&self, input_ids: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let device = input_ids.device();
        let mut generated = input_ids.clone();
        
        for _ in 0..max_new_tokens {
            let (logits, _) = self.forward(&generated)?;
            
            # Get last token logits
            let last_token_logits = logits.i((.., -1, ..))?;
            
            # Greedy: take argmax
            let next_token = last_token_logits.argmax(2)?.to_dtype(DType::U32)?;
            
            generated = Tensor::cat(&[&generated, &next_token], 1)?;
        }
        
        Ok(generated)
    }
}

# Multi-Token Prediction (MTP) head
# DeepSeek V3 uses MTP to predict multiple tokens ahead during training
pub struct MTPHead {
    linear: Linear,
    norm: candle_nn::RMSNorm,
}

impl MTPHead {
    pub fn new(vb: VarBuilder, config: &DeepSeekV3Config) -> Result<Self> {
        let linear = linear(vb.pp("mtp_linear"), config.hidden_size, config.hidden_size)?;
        let norm = candle_nn::rms_norm(vb.pp("mtp_norm"), config.hidden_size, config.rms_norm_eps)?;
        Ok(Self { linear, norm })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let output = self.linear.forward(hidden_states)?;
        self.norm.forward(&output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_model() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::new(DType::F32, &device);
        let config = DeepSeekV3Config::default();
        
        let model = DeepSeekV3::new(vb, &config)?;
        let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;
        
        let (logits, aux_loss) = model.forward(&input_ids)?;
        println!("Logits shape: {:?}", logits.shape());
        println!("Vocab size: {}", config.vocab_size);
        println!("Aux loss: {:?}", aux_loss);
        
        assert_eq!(logits.dims3()?.0, 1);
        assert_eq!(logits.dims3()?.1, 5);
        assert_eq!(logits.dims3()?.2, config.vocab_size);
        
        Ok(())
    }
}
