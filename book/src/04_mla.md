# Multi-Head Latent Attention (MLA)

## Overview

MLA compresses Key and Value vectors into a lower-dimensional latent space, dramatically reducing memory usage.

## Standard MHA vs MLA

```
Standard MHA:
- Store full KV: [batch, heads, seq, head_dim]
- Memory: O(seq_len × heads × head_dim)

MLA:
- Compress to latent: [batch, latent_dim]
- Memory: O(latent_dim) - constant!
```

## Implementation

```rust
pub struct MultiHeadLatentAttention {
    q_proj: Linear,
    kv_proj: Linear,  // Compressed!
    o_proj: Linear,
    latent_dim: usize,
}

impl MultiHeadLatentAttention {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Q: standard projection
        let q = self.q_proj.forward(x)?;
        
        // KV: compress to latent
        let kv = self.kv_proj.forward(x)?;
        let k_latent = kv.narrow(2, 0, self.latent_dim)?;
        let v_latent = kv.narrow(2, self.latent_dim, self.latent_dim)?;
        
        // Expand for attention (simplified)
        let k = k_latent.broadcastreshape(...)?;
        let v = v_latent.broadcastreshape(...)?;
        
        // Standard attention
        self.attention(q, k, v)
    }
}
```

## Benefits

1. **Memory**: 50%+ reduction in KV cache
2. **Long Context**: Can handle 128K tokens efficiently
3. **Latent + GQA**: Works with grouped query attention

## Comparison

| Method | KV Memory (128K) |
|--------|-----------------|
| Standard MHA | ~16 GB |
| MLA | ~2 GB |

This is crucial for DeepSeek V3.2's 128K context window!
