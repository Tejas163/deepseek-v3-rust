# Training Pipeline

## Overview

Training large language models requires careful setup of:
- Loss functions
- Optimizers
- Learning rate schedules
- Mixed precision

## Loss Function

### Language Modeling Loss

```rust
fn compute_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
    // Flatten for cross-entropy
    let logits = logits.reshape((batch * seq, vocab))?;
    let labels = labels.reshape((batch * seq,))?;
    
    // Cross-entropy
    candle_nn::loss::cross_entropy(&logits, &labels)
}
```

### Multi-Token Prediction Loss

```rust
fn mtp_loss(hidden: &Tensor, labels: &Tensor, model: &DeepSeekV3) -> Result<Tensor> {
    let mut total_loss = Tensor::zeros(1, hidden.device())?;
    
    for k in 1..num_predictions {
        let hs_k = hidden[:, :-k, :];
        let labels_k = labels[:, k:];
        
        let logits_k = model.lm_head(hs_k)?;
        let loss_k = cross_entropy(&logits_k, &labels_k)?;
        
        total_loss = total_loss.add(&loss_k)?;
    }
    
    Ok(total_loss / num_predictions as f32)
}
```

## Optimizer

```rust
use candle_optimizers::AdamW;

let optimizer = AdamW::new(
    lr,
    AdamW::Params {
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.01,
    },
)?;
```

## Training Loop

```rust
for epoch in 0..num_epochs {
    for batch in dataloader {
        // Forward
        let (logits, aux_loss) = model.forward(&input_ids)?;
        
        // Loss
        let loss = compute_loss(&logits, &labels)?;
        let total_loss = if let Some(aux) = aux_loss {
            loss.add(&aux.mul(0.001)?
        } else {
            loss
        };
        
        // Backward
        optimizer.backward_step(&total_loss)?;
        
        // Gradient clipping
        clip_grad_norm(model.parameters(), 1.0);
        
        optimizer.step()?;
    }
}
```

## FP8 Training

DeepSeek V3 uses FP8 mixed precision:
- Weights in FP32 (master)
- Activations in FP8
- Dynamic loss scaling

Benefits:
- 2-4x faster
- 50% less memory
- Similar accuracy
