---
tags: [flashcards]
source: 
summary: Hugging Face implementation of ViT
---

# `ViTLayer`
![[ViT An Image is Worth 16x16 Words Transformers for Image Recognition at Scale#Transformer Encoder]]
```python
class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config) 
        self.intermediate = ViTIntermediate(config) # first FC layer of intermediate_size
        self.output = ViTOutput(config) # second FC layer of hidden_size
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # ===== Equation 2 =========
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states
        # ===== END Equation 2 =========
        
        # ===== Equation 3 =========
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states) # Equation 3: LN
        layer_output = self.intermediate(layer_output) # Equation 3: first layer in MLP

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        # ===== END Equation 3 =========

        outputs = (layer_output,) + outputs

        return outputs
```