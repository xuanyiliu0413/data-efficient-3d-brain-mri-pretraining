import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x


class FeatureAdapter(nn.Module):





    def __init__(self, input_dim, output_dim, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens


        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, output_dim))

    def forward(self, features):






        B = features.shape[0]



        queries = self.query_tokens.expand(B, -1, -1)
        keys_values = features.unsqueeze(1)

        return queries, keys_values


class CrossAttention(nn.Module):




    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):








        B, N_q, D = query.shape
        N_kv = key_value.shape[1]


        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)


        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)


        x = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        x = self.out_proj(x)
        x = self.proj_dropout(x)

        return x


class MultiModalLateFusion(nn.Module):




    def __init__(self, embed_dim, proj_dim, num_heads=8, num_tokens=32, num_classes=2, fusion_type='gate'):
        super().__init__()
        self.fusion_type = fusion_type


        self.cross_attn_1to2 = CrossAttention(proj_dim, num_heads)

        self.cross_attn_2to1 = CrossAttention(proj_dim, num_heads)


        self.norm1 = nn.LayerNorm(proj_dim)
        self.norm2 = nn.LayerNorm(proj_dim)
        self.norm3 = nn.LayerNorm(proj_dim)
        self.norm4 = nn.LayerNorm(proj_dim)


        if fusion_type == 'concat':
            self.fusion = nn.Linear(proj_dim * 2, proj_dim)
        elif fusion_type == 'add':
            self.fusion = None
        elif fusion_type == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Tanh()
            )

        self.proj1 = ProjectionHead(
            embedding_dim=embed_dim,
            projection_dim=proj_dim,
            output_dim=proj_dim,
            dropout=0.1,
        )
        self.proj2 = ProjectionHead(
            embedding_dim=embed_dim,
            projection_dim=proj_dim,
            output_dim=proj_dim,
            dropout=0.1,
        )

        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, feat1, feat2):







        feat1, feat2 = self.proj1(feat1), self.proj2(feat2)


        feat1_attended = self.cross_attn_1to2(feat1, feat2)
        feat1_out = self.norm1(feat1 + self.norm3(feat1_attended))


        feat2_attended = self.cross_attn_2to1(feat2, feat1)
        feat2_out = self.norm2(feat2 + self.norm4(feat2_attended))


        feat1_pooled = feat1_out.mean(dim=1)
        feat2_pooled = feat2_out.mean(dim=1)


        if self.fusion_type == 'concat':
            fused = torch.cat([feat1_pooled, feat2_pooled], dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_type == 'add':
            fused = feat1_pooled + feat2_pooled
        elif self.fusion_type == 'gate':
            gate = self.gate(torch.cat([feat1_pooled, feat2_pooled], dim=-1))
            fused = gate * feat1_pooled + (1 - gate) * feat2_pooled

        out = self.classifier(fused)

        return out
