# Mixture of Experts (MoE)

## Concept

Mixture of Experts (MoE) enables conditional computation - only a subset of parameters are active for each input.

## Traditional FFN vs MoE

```rust
// Traditional FFN (dense)
fn ffn(x: &Tensor) -> Tensor {
    let up = linear(x, hidden, intermediate);
    let gate = linear(x, hidden, intermediate);
    up * sigmoid(gate)
}

// MoE: Multiple expert FFNs
fn moe(x: &Tensor, experts: &[Linear]) -> Tensor {
    let logits = gate(x); // [batch, num_experts]
    let (expert_ids, weights) = top_k(logits, k);
    
    // Get outputs from selected experts
    let mut output = zeros_like(x);
    for (id, weight) in zip(expert_ids, weights) {
        output += weight * experts[id].forward(x);
    }
    output
}
```

## DeepSeek V3 MoE Architecture

```rust
struct MoELayer {
    gate: Linear,           // Routes to experts
    experts: Vec<Linear>,   // 256 experts in full model
    shared_expert: Linear,  // Always active
}
```

### Shared Expert
- One expert that's always activated
- Processes all tokens
- Captures "general" knowledge

### Routed Experts
- Only top-k experts activated per token
- Each expert specializes in different aspects

## Auxiliary-Loss-Free Load Balancing

Traditional approach adds penalty:
```
loss = main_loss + λ * load_balance_loss
```

DeepSeek V3 uses dynamic bias:
```rust
// Add bias to gate logits based on expert usage
fn forward(&self, x: &Tensor) -> Tensor {
    let logits = self.gate(x);
    let biased_logits = logits + self.expert_bias;
    // ... routing with biased logits
}
```

This maintains load balance without sacrificing main objective!

## Implementation

```rust
pub struct MoELayer {
    gate: Linear,
    experts: Vec<Linear>,
    shared_expert: Linear,
    num_active_experts: usize,
}

impl MoELayer {
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        // 1. Get shared expert output (always active)
        let shared = self.shared_expert(x);
        
        // 2. Get routing logits
        let logits = self.gate(x);
        
        // 3. Select top-k experts
        let (expert_ids, weights) = logits.topk(self.num_active_experts, ...);
        
        // 4. Compute routed outputs
        let routed = ...;
        
        // 5. Combine
        shared + routed
    }
}
```

## Expert Capacity

Each expert has limited capacity:
```rust
let capacity = num_tokens * expert_capacity_factor / num_active_experts;
if token_count > capacity {
    // Overflow tokens go to fallback or shared expert
}
```
