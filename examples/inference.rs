// DeepSeek V3.2 Inference Example
// Run with: cargo run --example inference

use candle_core::{DType, Device, Result, Tensor};

use deepseek_v3::{DeepSeekV3, DeepSeekV3Config};

fn main() -> Result<()> {
    println!("DeepSeek V3.2 Inference Example");
    println!("===============================\n");

    let config = DeepSeekV3Config::default();
    let device = Device::Cpu;

    println!("Loading model...");
    let vb = candle_nn::VarBuilder::new(DType::F32, &device);
    let model = DeepSeekV3::new(vb, &config)?;
    println!("Model loaded!\n");

    // Simple tokenization (in practice, use proper tokenizer)
    let prompt = "Hello world";
    println!("Prompt: \"{}\"\n", prompt);

    // Convert to token IDs (simplified - just random for demo)
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;

    // Generate
    println!("Generating text...");
    let max_new_tokens = 20;
    let output_ids = model.generate(&input_ids, max_new_tokens)?;

    println!("Generated tokens: {:?}", output_ids.to_vec2::<u32>()?);

    // Decode (simplified)
    println!("\n[Demo: In practice, convert tokens back to text using tokenizer]");

    // === Beam Search Implementation ===
    println!("\n--- Beam Search Demo ---");
    let beam_width = 3;
    let beam_output = beam_search(&model, &input_ids, beam_width, max_new_tokens)?;
    println!("Beam search output: {:?}", beam_output);

    // === Greedy vs Sampling ===
    println!("\n--- Generation Methods Demo ---");
    println!("1. Greedy: Always pick highest probability token");
    println!("2. Beam Search: Keep top-k paths, select best sequence");
    println!("3. Temperature: Add randomness to probability distribution");
    println!("4. Top-k: Sample from top-k most likely tokens");
    println!("5. Top-p (Nucleus): Sample from smallest set with cumulative prob > p");

    Ok(())
}

// Beam Search Implementation from Scratch
fn beam_search(
    model: &DeepSeekV3,
    input_ids: &Tensor,
    beam_width: usize,
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    let device = input_ids.device();

    // Initialize with input
    let mut sequences = vec![input_ids.clone()];
    let mut scores = vec![0.0f32; beam_width];

    for _ in 0..max_new_tokens {
        let mut candidates = Vec::new();

        for (seq, score) in sequences.iter().zip(scores.iter()) {
            let (logits, _) = model.forward(seq)?;

            // Get last token logits
            let last_logits = logits.i((.., -1, ..))?;

            // Get top-k tokens
            let (topk_ids, topk_probs) = last_logits.topk(beam_width, 2, true, true)?;

            // Expand each beam
            for i in 0..beam_width {
                let next_id = topk_ids
                    .flatten_all()?
                    .to_vec1::<u32>()?
                    .get(i)
                    .copied()
                    .unwrap_or(0);
                let next_prob = topk_probs
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .get(i)
                    .copied()
                    .unwrap_or(0.0);

                let new_seq = Tensor::cat(seq, &Tensor::new(&[[next_id]], device)?, 1)?;
                candidates.push((new_seq, score + next_prob));
            }
        }

        // Select top beam_width candidates
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sequences = candidates.iter().map(|(s, _)| s.clone()).collect();
        scores = candidates
            .iter()
            .map(|(_, s)| *s)
            .take(beam_width)
            .collect();
    }

    // Return best sequence
    let best_idx = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_seq = sequences
        .get(best_idx)
        .ok_or(candle_core::Error::Msg("No sequence found".into()))?;

    Ok(best_seq.to_vec2::<u32>()?.remove(0))
}
