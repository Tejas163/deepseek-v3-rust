use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSeekV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_experts: usize,
    pub num_active_experts: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub expert_capacity_factor: f32,
}

impl Default for DeepSeekV3Config {
    fn default() -> Self {
        // Simplified version for learning (vs full 671B model)
        Self {
            vocab_size: 50000,
            hidden_size: 512,        // vs 7168
            intermediate_size: 1376, // ~2.5x hidden
            num_hidden_layers: 4,    // vs 61
            num_attention_heads: 8,  // vs 128
            num_kv_heads: 1,        # GQA - Grouped Query Attention
            num_experts: 8,         # vs 256
            num_active_experts: 2,   # vs 8-9
            max_position_embeddings: 512, # vs 128K
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            expert_capacity_factor: 1.25,
        }
    }
}

impl DeepSeekV3Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    pub fn estimate_params(&self) -> usize {
        // Embedding params
        let embed_params = self.vocab_size * self.hidden_size * 2;
        
        // Attention params per layer
        let qkv_params = self.hidden_size * self.hidden_size * 3;
        let out_params = self.hidden_size * self.hidden_size;
        
        // MoE params per layer
        let expert_params = self.intermediate_size * self.hidden_size * self.num_experts;
        let gate_params = self.hidden_size * self.num_experts;
        
        let per_layer = qkv_params + out_params + expert_params + gate_params;
        embed_params + (per_layer * self.num_hidden_layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = DeepSeekV3Config::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.num_experts, 8);
    }
    
    #[test]
    fn test_estimate_params() {
        let config = DeepSeekV3Config::default();
        let params = config.estimate_params();
        println!("Estimated params: {}M", params / 1_000_000);
        assert!(params > 0);
    }
}
