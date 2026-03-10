# Inference & Generation

## Generation Methods

### 1. Greedy Decoding

Always pick the highest probability token:

```rust
fn greedy_decode(model: &DeepSeekV3, input_ids: &Tensor, max_new: usize) -> Result<Tensor> {
    let mut generated = input_ids.clone();
    
    for _ in 0..max_new {
        let (logits, _) = model.forward(&generated)?;
        let next_token = logits.argmax(2)?;  // argmax
        generated = Tensor::cat(&[&generated, &next_token], 1)?;
    }
    
    Ok(generated)
}
```

### 2. Beam Search

Keep top-k paths, select best:

```rust
fn beam_search(model: &DeepSeekV3, input_ids: &Tensor, beam_width: usize, max_new: usize) -> Result<Tensor> {
    let mut sequences = vec![input_ids.clone()];
    let mut scores = vec![0.0f32; beam_width];
    
    for _ in 0..max_new {
        let mut candidates = Vec::new();
        
        for (seq, score) in sequences.iter().zip(scores.iter()) {
            let (logits, _) = model.forward(seq)?;
            let (topk_ids, topk_probs) = logits.topk(beam_width, 2, true, true)?;
            
            for i in 0..beam_width {
                let new_seq = Tensor::cat(seq, &topk_ids.narrow(2, i, 1)?, 1)?;
                candidates.push((new_seq, score + topk_probs.narrow(2, i, 1)?));
            }
        }
        
        // Keep top beams
        candidates.sort_by(|a, b| b.1.cmp(&a.1));
        sequences = candidates.iter().map(|(s, _)| s.clone()).collect();
        scores = candidates.iter().map(|(_, s)| *s).take(beam_width).collect();
    }
    
    Ok(sequences[0].clone())
}
```

### 3. Sampling Methods

```rust
// Temperature sampling
fn temperature_sample(logits: &Tensor, temperature: f32) -> Result<Tensor> {
    if temperature == 0.0 {
        return logits.argmax(2);
    }
    
    let scaled = logits.div(temperature)?;
    let probs = softmax(&scaled, 2)?;
    // Sample from distribution
    probs.argmax(2)
}

// Top-k sampling
fn topk_sample(logits: &Tensor, k: usize) -> Result<Tensor> {
    let (topk, _) = logits.topk(k, 2, true, true)?;
    let mask = logits.lt(&topk.narrow(2, k-1, 1)?)?;
    let masked = logits.where(&mask, &Tensor::new(f32::NEG_INFINITY, logits.device())?)?;
    softmax(&masked, 2)?.argmax(2)
}
```

## Comparison

| Method | Quality | Speed | Diversity |
|--------|---------|-------|-----------|
| Greedy | Good | Fastest | None |
| Beam Search | Best | Slow | Low |
| Temperature | Varied | Fast | High |
| Top-k | Good | Fast | Medium |
| Top-p | Good | Fast | High |

## Model Quantization

Reduce model size for deployment:

```rust
// INT8 quantization (simplified)
fn quantize_int8(weights: &Tensor) -> Result<(Tensor, Tensor)> {
    // Get scale
    let max = weights.abs().max()?;
    let scale = max / 127.0;
    
    // Quantize
    let quantized = weights.div(scale).round().to_dtype(DType::I8)?;
    
    Ok((quantized, scale))
}
```
