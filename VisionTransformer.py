import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = rearrange(
            qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, h=self.n_heads
        )  # (qkv, n_samples, n_heads, n_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # split q, k, v from dim 0

        dot_prod = torch.einsum(
            'bhid,bhjd->bhij', q, k
        ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dot_prod.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhij,bhjd->bhid', attn, v)  # (n_samples, n_heads, n_patches + 1, head_dim)
        x = rearrange(x, 'b h t d -> b t (h d)')  # (n_samples, n_patches + 1, dim)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=hidden_features, out_features=dim
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(self.patch_embed.n_patches**0.5))
        x = self.decoder(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        return x
    
# -------------------------------------------------------------------------------------------------------------------- # FIRST VERSION

# class TransformerEncoder(nn.Module):
#     def __init__(
#         self,
#         img_size=256,
#         patch_size=16,
#         in_chans=1,
#         embed_dim=768,
#         depth=12,
#         n_heads=12,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         p=0.,
#         attn_p=0.,
#     ):
#         super().__init__()

#         self.patch_embed = PatchEmbed(
#             img_size=img_size,
#             patch_size=patch_size,
#             in_chans=in_chans,
#             embed_dim=embed_dim,
#         )
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, self.patch_embed.n_patches, embed_dim)
#         )
#         self.pos_drop = nn.Dropout(p=p)

#         self.blocks = nn.ModuleList(
#             [
#                 Block(
#                     dim=embed_dim,
#                     n_heads=n_heads,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     p=p,
#                     attn_p=attn_p,
#                 )
#                 for _ in range(depth)
#             ]
#         )

#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)

#         for block in self.blocks:
#             x = block(x)

#         x = self.norm(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=int(self.patch_embed.n_patches**0.5))

#         return x

# class TransformerUNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = TransformerEncoder(
#             img_size=config['img_size'],
#             patch_size=config['patch_size'],
#             in_chans=config['in_channels'],
#             embed_dim=config['embed_dim'],
#             depth=config['depth'],
#             n_heads=config['n_heads'],
#             mlp_ratio=config['mlp_ratio'],
#             qkv_bias=config['qkv_bias'],
#             p=config['p'],
#             attn_p=config['attn_p']
#         )
        
#         self.pretrained_model = smp.create_model(
#             arch=config['arch'],
#             encoder_name=config['encoder_name'],
#             encoder_weights=config['encoder_weights'],
#             in_channels=config['in_channels'],
#             classes=config['classes']
#         )
        
#         # Remove the segmentation head from the pretrained model
#         self.pretrained_model.segmentation_head = nn.Identity()
        
#         decoder_channels = (256, 128, 64, 32, 16)
#         encoder_channels = self.pretrained_model.encoder.out_channels
        
#         self.decoder = nn.ModuleList()
#         for i in range(len(decoder_channels)):
#             in_ch = encoder_channels[-i-1] if i == 0 else decoder_channels[i-1]
#             out_ch = decoder_channels[i]
#             self.decoder.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
#             self.decoder.append(nn.BatchNorm2d(out_ch))
#             self.decoder.append(nn.ReLU(inplace=True))
#             if i < len(decoder_channels) - 1:
#                 self.decoder.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
#                 self.decoder.append(nn.BatchNorm2d(out_ch))
#                 self.decoder.append(nn.ReLU(inplace=True))
#                 self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
#         self.fusion = nn.Sequential(
#             nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
#             nn.BatchNorm2d(encoder_channels[-1]),
#             nn.ReLU(inplace=True)
#         )
        
#         self.segmentation_head = nn.Conv2d(decoder_channels[-1], config['classes'], kernel_size=1)

#     def forward(self, x):
#         transformer_features = self.encoder(x)
#         pretrained_features = self.pretrained_model.encoder(x)
        
#         transformer_features = self.fusion(transformer_features)
#         features = transformer_features + pretrained_features[-1]
        
#         decoder_output = features
#         for layer in self.decoder:
#             decoder_output = layer(decoder_output)
        
#         masks = self.segmentation_head(decoder_output)
#         return masks

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size, patch_size, in_chans, embed_dim):
#         super().__init__()
#         self.n_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
#     def forward(self, x):
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x

# class Block(nn.Module):
#     def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, p=0., attn_p=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_p, bias=qkv_bias)
#         self.drop_path = nn.Identity()
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(p),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(p)
#         )
    
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
    
    
    
    
# -------------------------------------------------------------------------------------------------------------------- # SECOND VERSION


# class PatchEmbed(nn.Module):
#     def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.n_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
#     def forward(self, x):
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
#         super().__init__()
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
#         self.pos_drop = nn.Dropout(p=p)

#         self.blocks = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=int(embed_dim * mlp_ratio), dropout=p, activation='gelu')
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.norm(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=int(self.patch_embed.n_patches**0.5))
#         return x

# class TransformerUNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = TransformerEncoder(
#             img_size=config['img_size'],
#             patch_size=config['patch_size'],
#             in_chans=config['in_channels'],
#             embed_dim=config['embed_dim'],
#             depth=config['depth'],
#             n_heads=config['n_heads'],
#             mlp_ratio=config['mlp_ratio'],
#             qkv_bias=config['qkv_bias'],
#             p=config['p'],
#             attn_p=config['attn_p']
#         )
        
#         self.pretrained_model = smp.create_model(
#             arch=config['arch'],
#             encoder_name=config['encoder_name'],
#             encoder_weights=config['encoder_weights'],
#             in_channels=config['in_channels'],
#             classes=config['classes']
#         )
#         self.pretrained_model.segmentation_head = nn.Identity()
        
#         decoder_channels = (256, 128, 64, 32, 16)
#         encoder_channels = self.pretrained_model.encoder.out_channels
        
#         self.decoder = nn.ModuleList()
#         for i in range(len(decoder_channels)):
#             in_ch = encoder_channels[-i-1] if i == 0 else decoder_channels[i-1]
#             out_ch = decoder_channels[i]
#             self.decoder.append(nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if i < len(decoder_channels) - 1 else nn.Identity()
#             ))
        
#         self.fusion = nn.Sequential(
#             nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
#             nn.BatchNorm2d(encoder_channels[-1]),
#             nn.ReLU(inplace=True)
#         )
        
#         self.segmentation_head = nn.Conv2d(decoder_channels[-1], config['classes'], kernel_size=1)

#     def forward(self, x):
#         transformer_features = self.encoder(x)
#         pretrained_features = self.pretrained_model.encoder(x)
        
#         transformer_features = self.fusion(transformer_features)
#         features = transformer_features + pretrained_features[-1]
        
#         decoder_output = features
#         for layer in self.decoder:
#             decoder_output = layer(decoder_output)
        
#         masks = self.segmentation_head(decoder_output)
#         return masks

# -------------------------------------------------------------------------------------------------------------------- # HYBRID MODEL
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# import segmentation_models_pytorch as smp

# class PatchEmbed(nn.Module):
#     def __init__(self, img_size, patch_size, in_chans, embed_dim):
#         super().__init__()
#         self.n_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
#     def forward(self, x):
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)
#         return x

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.global_avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class TransformerBlock(nn.Module):
#     def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, p=0., attn_p=0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_p, bias=qkv_bias)
#         self.drop_path = nn.Identity()
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, int(dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(p),
#             nn.Linear(int(dim * mlp_ratio), dim),
#             nn.Dropout(p)
#         )

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
#         super().__init__()
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
#         self.pos_drop = nn.Dropout(p=p)
#         self.blocks = nn.ModuleList([
#             TransformerBlock(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

#     def forward(self, x):
#         x = self.patch_embed(x)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.norm(x)
#         x = rearrange(x, 'b (h w) c -> b c h w', h=int(self.patch_embed.n_patches**0.5))
#         return x

# class TransformerUNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = TransformerEncoder(
#             img_size=config['img_size'],
#             patch_size=config['patch_size'],
#             in_chans=config['in_channels'],
#             embed_dim=config['embed_dim'],
#             depth=config['depth'],
#             n_heads=config['n_heads'],
#             mlp_ratio=config['mlp_ratio'],
#             qkv_bias=config['qkv_bias'],
#             p=config['p'],
#             attn_p=config['attn_p']
#         )
        
#         self.pretrained_model = smp.create_model(
#             arch=config['arch'],
#             encoder_name=config['encoder_name'],
#             encoder_weights=config['encoder_weights'],
#             in_channels=config['in_channels'],
#             classes=config['classes']
#         )
#         self.pretrained_model.segmentation_head = nn.Identity()
        
#         decoder_channels = (256, 128, 64, 32, 16)
#         encoder_channels = self.pretrained_model.encoder.out_channels
        
#         self.decoder = nn.ModuleList()
#         for i in range(len(decoder_channels)):
#             in_ch = encoder_channels[-i-1] if i == 0 else decoder_channels[i-1]
#             out_ch = decoder_channels[i]
#             self.decoder.append(nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 SEBlock(out_ch),
#                 nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_ch),
#                 nn.ReLU(inplace=True),
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if i < len(decoder_channels) - 1 else nn.Identity()
#             ))
        
#         self.fusion = nn.Sequential(
#             nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
#             nn.BatchNorm2d(encoder_channels[-1]),
#             nn.ReLU(inplace=True)
#         )
        
#         self.segmentation_head = nn.Conv2d(decoder_channels[-1], config['classes'], kernel_size=1)

#     def forward(self, x):
#         transformer_features = self.encoder(x)
#         pretrained_features = self.pretrained_model.encoder(x)
        
#         transformer_features = self.fusion(transformer_features)
#         features = transformer_features + pretrained_features[-1]
        
#         decoder_output = features
#         for layer in self.decoder:
#             decoder_output = layer(decoder_output)
        
#         masks = self.segmentation_head(decoder_output)
#         return masks
    
    
    
    
    #-------------------------------------------------------------------------------------------------------------------- # FIRST VERSION ENHANCED
    
    
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(self.patch_embed.n_patches**0.5))

        return x
    
    
class TransformerUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            in_chans=config['in_channels'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            n_heads=config['n_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config['qkv_bias'],
            p=config['p'],
            attn_p=config['attn_p']
        )
        
        self.pretrained_model = smp.create_model(
            arch=config['arch'],
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            in_channels=config['in_channels'],
            classes=config['classes']
        )
        
        # Remove the segmentation head from the pretrained model
        self.pretrained_model.segmentation_head = nn.Identity()
        
        self.decoder_channels = (256, 128, 64, 32, 16)
        encoder_channels = self.pretrained_model.encoder.out_channels
        
        self.decoder = nn.ModuleList()
        for i in range(len(self.decoder_channels)):
            in_ch = encoder_channels[-i-1] if i == 0 else self.decoder_channels[i]
            out_ch = self.decoder_channels[i]
            self.decoder.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.decoder.append(nn.ReLU(inplace=True))
            if i < len(self.decoder_channels) - 1:
                self.decoder.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
                self.decoder.append(nn.ReLU(inplace=True))
                self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
                self.decoder.append(nn.Conv2d(self.decoder_channels[i+1], out_ch, kernel_size=1, stride=1, padding=0))
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.fusion_attn = nn.MultiheadAttention(encoder_channels[-1], num_heads=8, dropout=0.1)
        
        self.segmentation_head = nn.Conv2d(self.decoder_channels[-1], config['classes'], kernel_size=1)

    def forward(self, x):
        transformer_features = self.encoder(x)
        pretrained_features = self.pretrained_model.encoder(x)
        
        transformer_features = self.fusion_conv(transformer_features)
        batch_size, channels, height, width = transformer_features.size()
        transformer_features = transformer_features.view(batch_size, channels, height * width).permute(2, 0, 1)
        transformer_features, _ = self.fusion_attn(transformer_features, transformer_features, transformer_features)
        transformer_features = transformer_features.permute(1, 2, 0).view(batch_size, channels, height, width)
        features = transformer_features + pretrained_features[-1]
        
        decoder_output = features
        for i, layer in enumerate(self.decoder):
            print(f"Layer {i}: {layer}")
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
                if i == len(self.decoder) - 1:
                    decoder_output = layer(decoder_output)
                else:
                    print(f'decoder_output shape {len(decoder_output)} decoder_channels shape {len(self.decoder_channels)}')
                    decoder_output = nn.Conv2d(decoder_output.size(1), self.decoder_channels[i+1], kernel_size=1, stride=1, padding=0)(decoder_output)
                    decoder_output = layer(decoder_output)
            else:
                decoder_output = layer(decoder_output)
                if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                    decoder_output = decoder_output[:, :self.decoder_channels[i], :, :]
        
        masks = self.segmentation_head(decoder_output)
        return masks
    
    
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=False, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=attn_p, bias=qkv_bias)
        self.drop_path = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(p)
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x