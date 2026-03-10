use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

mod config;
mod embeddings;
mod mla;
mod moe;
mod dsa;
mod block;
mod model;

pub use config::DeepSeekV3Config;
pub use embeddings::TokenEmbeddings;
pub use mla::MultiHeadLatentAttention;
pub use moe::MoELayer;
pub use dsa::DeepSeekSparseAttention;
pub use block::DeepSeekBlock;
pub use model::DeepSeekV3;

fn main() -> Result<()> {
    println!("DeepSeek V3.2 - Rust Implementation (Candle)");
    println!("============================================");
    
    let config = DeepSeekV3Config::default();
    println!("Config: {:?}", config);
    
    let device = Device::Cpu;
    println!("Device: {:?}", device);
    
    // Quick sanity check - create embeddings
    let vb = VarBuilder::new(DType::F32, &device);
    let embeddings = TokenEmbeddings::new(&vb, config.vocab_size, config.hidden_size)?;
    println!("Embeddings created successfully!");
    
    // Test forward pass
    let input_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device)?;
    let output = embeddings.forward(&input_ids)?;
    println!("Input shape: {:?}", input_ids.shape());
    println!("Output shape: {:?}", output.shape());
    
    println!("\n✅ Basic implementation working!");
    println!("\nNext steps:");
    println!("  - Run: cargo run --example train");
    println!("  - Run: cargo run --example inference");
    
    Ok(())
}
