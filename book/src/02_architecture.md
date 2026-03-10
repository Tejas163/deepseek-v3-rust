# DeepSeek V3.2 Architecture

## Overview

DeepSeek V3.2 represents a breakthrough in efficient LLM architecture through several key innovations.

## Key Components

### 1. Multi-Head Latent Attention (MLA)

MLA compresses the Key-Value cache into a lower-dimensional latent space:

```
Traditional MHA:
- Store full KV cache: [batch, num_heads, seq_len, head_dim]

MLA:
- Compress to latent: [batch, latent_dim]
- Expand when needed
- Saves significant memory for long contexts
```

### 2. Mixture of Experts (MoE)

Instead of one large FFN, MoE has multiple specialized experts:

```
Input → Router → Select Top-k Experts → Combine Outputs
```

DeepSeek V3.2 has:
- 256 total experts
- 8-9 activated per token
- Plus 1 shared expert (always active)

### 3. DeepSeek Sparse Attention (DSA)

Reduces attention complexity from O(L²) to O(L·k):

```
Standard Attention: Attend to ALL past tokens
DSA: Attend to only k most relevant tokens per position

For 128K context with k=512:
- Standard: 128K² = 16B operations
- DSA: 128K × 512 = 65M operations
```

### 4. Multi-Token Prediction (MTP)

Predicts multiple tokens ahead during training:

```
Traditional: predict token t+1
MTP: predict tokens t+1, t+2, t+3 simultaneously
```

### 5. Auxiliary-Loss-Free Load Balancing

Instead of adding a penalty term to loss, DeepSeek uses:
- Dynamic bias adjustment based on expert affinity
- Maintains load balance without hurting main objective

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | 671B |
| Activated Parameters | 37B |
| Number of Experts | 256 |
| Active Experts | 8-9 |
| Context Length | 128K |
| Hidden Size | 7168 |
| Number of Layers | 61 |
| Attention Heads | 128 |
