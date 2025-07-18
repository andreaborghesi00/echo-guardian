import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import segmentation_models_pytorch as smp
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
    
    
    
# #-------------------------------------------------------------------------------------------------------------------- # SECOND VERSION


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
        
#         self.decoder_channels = (256, 128, 64, 32, 16)
#         encoder_channels = self.pretrained_model.encoder.out_channels
        
#         self.decoder = nn.ModuleList()
#         for i in range(len(self.decoder_channels)):
#             in_ch = encoder_channels[-i-1] if i == 0 else self.decoder_channels[i]
#             out_ch = self.decoder_channels[i]
#             self.decoder.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
#             self.decoder.append(nn.ReLU(inplace=True))
#             if i < len(self.decoder_channels) - 1:
#                 self.decoder.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
#                 self.decoder.append(nn.ReLU(inplace=True))
#                 self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
#                 self.decoder.append(nn.Conv2d(self.decoder_channels[i+1], out_ch, kernel_size=1, stride=1, padding=0))
        
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
#             nn.BatchNorm2d(encoder_channels[-1]),
#             nn.ReLU(inplace=True)
#         )
#         self.fusion_attn = nn.MultiheadAttention(encoder_channels[-1], num_heads=8, dropout=0.1)
        
#         self.segmentation_head = nn.Conv2d(self.decoder_channels[-1], config['classes'], kernel_size=1)

#     def forward(self, x):
#         transformer_features = self.encoder(x)
#         pretrained_features = self.pretrained_model.encoder(x)
        
#         transformer_features = self.fusion_conv(transformer_features)
#         batch_size, channels, height, width = transformer_features.size()
#         transformer_features = transformer_features.view(batch_size, channels, height * width).permute(2, 0, 1)
#         transformer_features, _ = self.fusion_attn(transformer_features, transformer_features, transformer_features)
#         transformer_features = transformer_features.permute(1, 2, 0).view(batch_size, channels, height, width)
#         features = transformer_features + pretrained_features[-1]
        
#         decoder_output = features
#         for i, layer in enumerate(self.decoder):
#             print(f"Layer {i}: {layer}")
#             if isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1):
#                 if i == len(self.decoder) - 1:
#                     decoder_output = layer(decoder_output)
#                 else:
#                     decoder_output = nn.Conv2d(decoder_output.size(1), self.decoder_channels[i+1], kernel_size=1, stride=1, padding=0)(decoder_output)
#                     decoder_output = layer(decoder_output)
#             else:
#                 decoder_output = layer(decoder_output)
#                 if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
#                     decoder_output = decoder_output[:, :self.decoder_channels[i], :, :]
        
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



# -------------------------------------------------------------------------------------------------------------------- # FIRST VERSION ENHANCED PT2



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
        
        decoder_channels = (256, 128, 64, 32, 16)
        encoder_channels = self.pretrained_model.encoder.out_channels
        
        self.decoder = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-i-1] if i == 0 else decoder_channels[i-1]
            out_ch = decoder_channels[i]
            self.decoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            if i < len(decoder_channels) - 1:
                self.decoder.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.attention_blocks.append(AttentionBlock(out_ch, encoder_channels[-i-2] if i < len(decoder_channels)-1 else out_ch))
        
        self.fusion = nn.Sequential(
            nn.Conv2d(config['embed_dim'], encoder_channels[-1], kernel_size=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], config['classes'], kernel_size=1)

    def forward(self, x):
        transformer_features = self.encoder(x)
        pretrained_features = self.pretrained_model.encoder(x)
        
        transformer_features = self.fusion(transformer_features)
        transformer_features = F.interpolate(transformer_features, size=pretrained_features[-1].shape[2:], mode='bilinear', align_corners=False)
        features = transformer_features + pretrained_features[-1]
        
        decoder_output = features
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Sequential):
                decoder_output = layer(decoder_output)
            else:
                decoder_output = layer(decoder_output)
            if i % 2 == 1 and i // 2 < len(self.attention_blocks):  # Apply attention every two Conv-BN-ReLU blocks
                decoder_output = self.attention_blocks[i // 2](decoder_output, pretrained_features[-(i // 2 + 2)])
        
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

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, g_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, g):
        g1 = self.W_g(g)
        g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
