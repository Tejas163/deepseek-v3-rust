# DeepSeek V3.2 Implementation in Rust

This is a comprehensive implementation of DeepSeek V3.2 architecture using the Candle ML framework in Rust.

## What is DeepSeek V3.2?

DeepSeek V3.2 is a state-of-the-art 671B parameter Mixture-of-Experts (MoE) language model featuring:

- **671B total parameters** with only **37B activated** per token
- **128K context window**
- Key architectural innovations:
  - Multi-Head Latent Attention (MLA)
  - DeepSeek Sparse Attention (DSA)
  - Multi-Token Prediction (MTP)
  - Auxiliary-loss-free load balancing

## Why Rust?

Rust provides several advantages for ML implementation:
- Memory safety without garbage collection
- Zero-cost abstractions
- High performance (comparable to C++)
- Can run without Python overhead

## Project Structure

```
rust/
├── src/
│   ├── main.rs          # Entry point
│   ├── config.rs        # Model configuration
│   ├── embeddings.rs    # Token & positional embeddings
│   ├── mla.rs           # Multi-Head Latent Attention
│   ├── moe.rs          # Mixture of Experts layer
│   ├── dsa.rs           # DeepSeek Sparse Attention
│   ├── block.rs         # Complete transformer block
│   └── model.rs         # Full model
├── examples/
│   ├── train.rs         # Training example
│   └── inference.rs     # Inference example
└── book/                # Documentation
```

## Quick Start

```bash
# Build
cargo build

# Run inference example
cargo run --example inference

# Run training example  
cargo run --example train
```

## Simplified Model

For learning purposes, we implement a simplified version:

| Component | Full Model | Our Implementation |
|-----------|-----------|-------------------|
| Hidden Size | 7168 | 512 |
| Layers | 61 | 4 |
| Experts | 256 | 8 |
| Active Experts | 8-9 | 2 |
| Context | 128K | 512 |

This reduces ~700M parameters to ~100M, fitting on consumer GPU.
