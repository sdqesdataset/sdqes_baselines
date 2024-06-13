import torch
from torch import nn
from x_transformers.x_transformers import AttentionLayers, AbsolutePositionalEmbedding, exists, default
from einops import rearrange, repeat
from torch import nn, einsum


class TransformerPooler(nn.Module):
    def __init__(
        self,
        max_seq_len=16,
        transformer_dim=512,
        transformer_ff_dim=None,
        transformer_input_dim=512,
        transformer_depth=1,
        transformer_heads=2,
        transformer_emb_dropout=0.0,
        post_emb_norm=True,
        use_abs_pos_emb=True,
        **kwargs,
    ):
        super().__init__()

        self.use_abs_pos_emb = use_abs_pos_emb

        transformer_ff_dim = default(transformer_ff_dim, transformer_dim * 2)   # default to 2x transformer_dim if not specified

        # projection layers into transformer dimension (lazy init to not have to pass in input_dim)
        self.x_proj = nn.Linear(in_features=transformer_input_dim, out_features=transformer_dim)
        
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(
                dim=transformer_dim,
                max_seq_len=max_seq_len,
            )
        else:
            raise NotImplementedError

        # class embeddings (initialized with normal distribution)
        self.csl_emb = nn.Parameter(torch.randn(1, 1, transformer_dim), requires_grad=True)
        self.emb_dropout = nn.Dropout(transformer_emb_dropout)
        # post embedding normalization
        self.post_emb_norm = nn.LayerNorm(transformer_dim) if post_emb_norm else nn.Identity()

        # LxL transformer layers (L = max_seq_len)
        self.transformer = AttentionLayers(
            dim=transformer_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            ff_mult=transformer_ff_dim / transformer_dim,
            pre_norm=True,
        )

        # post transformer normalization
        self.norm = nn.LayerNorm(transformer_dim)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs (tensor) : [batch, seq_len, input_dim]
        Returns:
            output (tensor): [batch, pooling_dim]
        """
        cls_emb = self.csl_emb
        
        #########################
        # project inputs to the transformer dimension
        inputs = self.x_proj(inputs)                                             # => [batch, seq_len, transformer_dim]

        #########################
        # add positional embeddings to sequences
        pos_emb = self.pos_emb(inputs, pos=None)
        inputs += pos_emb

        #########################
        # append cls embedding sequences to goal embedding / query
        cls_emb = repeat(cls_emb, '1 1 d -> b 1 d', b=inputs.shape[0])
        x = torch.cat([cls_emb, inputs], dim=-2)                                   # => [batch, 1 + seq_len, transformer_dim]

        #########################
        # post embedding norm, purportedly leads to greater stabilization
        x = self.post_emb_norm(x)

        #########################
        # embedding dropout
        x = self.emb_dropout(x)

        #########################
        # use transformer layers to get a latent representation of the input
        x = self.transformer(x, mask=None)                       # => [batch, 1 + seq_len, transformer_dim]

        return x[:,0]
