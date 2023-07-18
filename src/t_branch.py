from functools import partial
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention

class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx

class Block (nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=DualTransformerBlock, Mlp_block=Mlp
                 , init_values=1e-4):
        super ().__init__ ()
        self.norm1 = norm_layer (dim)
        self.attn = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath (drop_path) if drop_path > 0. else nn.Identity ()
        self.norm2 = norm_layer (dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path (self.attn (self.norm1 (x)))
        x = x + self.drop_path (self.mlp (self.norm2 (x)))
        return x

class Layer_scale_init_Block (nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=DualTransformerBlock, Mlp_block=Mlp
                 , init_values=1e-4):
        super ().__init__ ()
        self.norm1 = norm_layer (dim)
        self.attn = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath (drop_path) if drop_path > 0. else nn.Identity ()
        self.norm2 = norm_layer (dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path (self.gamma_1 * self.attn (self.norm1 (x)))
        x = x + self.drop_path (self.gamma_2 * self.mlp (self.norm2 (x)))
        return x

class Layer_scale_init_Block_paralx2 (nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=DualTransformerBlock, Mlp_block=Mlp
                 , init_values=1e-4):
        super ().__init__ ()
        self.norm1 = norm_layer (dim)
        self.norm11 = norm_layer (dim)
        self.attn = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath (drop_path) if drop_path > 0. else nn.Identity ()
        self.norm2 = norm_layer (dim)
        self.norm21 = norm_layer (dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path (self.gamma_1 * self.attn (self.norm1 (x))) + self.drop_path (
            self.gamma_1_1 * self.attn1 (self.norm11 (x)))
        x = x + self.drop_path (self.gamma_2 * self.mlp (self.norm2 (x))) + self.drop_path (
            self.gamma_2_1 * self.mlp1 (self.norm21 (x)))
        return x

class Block_paralx2 (nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=DualTransformerBlock, Mlp_block=Mlp
                 , init_values=1e-4):
        super ().__init__ ()
        self.norm1 = norm_layer (dim)
        self.norm11 = norm_layer (dim)
        self.attn = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block (
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath (drop_path) if drop_path > 0. else nn.Identity ()
        self.norm2 = norm_layer (dim)
        self.norm21 = norm_layer (dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block (in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path (self.attn (self.norm1 (x))) + self.drop_path (self.attn1 (self.norm11 (x)))
        x = x + self.drop_path (self.mlp (self.norm2 (x))) + self.drop_path (self.mlp1 (self.norm21 (x)))
        return x

class hMLP_stem (nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm):
        super ().__init__ ()
        img_size = to_2tuple (img_size)
        patch_size = to_2tuple (patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential (*[nn.Conv2d (in_chans, embed_dim // 4, kernel_size=4, stride=4),
                                           norm_layer (embed_dim // 4),
                                           nn.GELU (),
                                           nn.Conv2d (embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                                           norm_layer (embed_dim // 4),
                                           nn.GELU (),
                                           nn.Conv2d (embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                                           norm_layer (embed_dim),
                                           ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj (x).flatten (2).transpose (1, 2)
        return x

class vit_models (nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=DualTransformerBlock, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, **kwargs):
        super ().__init__ ()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer (
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter (torch.zeros (1, 1, embed_dim))

        self.pos_embed = nn.Parameter (torch.zeros (1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range (depth)]
        self.blocks = nn.ModuleList ([
            block_layers (
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range (depth)])

        self.norm = norm_layer (embed_dim)

        self.feature_info = [dict (num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear (embed_dim, num_classes) if num_classes > 0 else nn.Identity ()

        trunc_normal_ (self.pos_embed, std=.02)
        trunc_normal_ (self.cls_token, std=.02)
        self.apply (self._init_weights)

    def _init_weights(self, m):
        if isinstance (m, nn.Linear):
            trunc_normal_ (m.weight, std=.02)
            if isinstance (m, nn.Linear) and m.bias is not None:
                nn.init.constant_ (m.bias, 0)
        elif isinstance (m, nn.LayerNorm):
            nn.init.constant_ (m.bias, 0)
            nn.init.constant_ (m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len (self.blocks)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear (self.embed_dim, num_classes) if num_classes > 0 else nn.Identity ()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed (x)

        cls_tokens = self.cls_token.expand (B, -1, -1)

        x = x + self.pos_embed

        x = torch.cat ((cls_tokens, x), dim=1)

        for i, blk in enumerate (self.blocks):
            x = blk (x)

        x = self.norm (x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features (x)

        if self.dropout_rate:
            x = F.dropout (x, p=float (self.dropout_rate), training=self.training)
        x = self.head (x)

        return x

@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model

@register_model
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg ()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_' + str (img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url (
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict (checkpoint["model"])

    return model

@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg ()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_' + str (img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url (
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict (checkpoint["model"])
    return model

@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_' + str (img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url (
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict (checkpoint["model"])
    return model

@register_model
def deit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_' + str (img_size) + '_'
        if pretrained_21k:
            name += '21k.pth'
        else:
            name += '1k.pth'

        checkpoint = torch.hub.load_state_dict_from_url (
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict (checkpoint["model"])
    return model

@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_' + str (img_size) + '_'
        if pretrained_21k:
            name += '21k_v1.pth'
        else:
            name += '1k_v1.pth'

        checkpoint = torch.hub.load_state_dict_from_url (
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict (checkpoint["model"])
    return model

@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model

@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model

@register_model
def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Block_paral_LS, **kwargs)

    return model

@register_model
def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Block_paral_LS, **kwargs)
    return model

@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    # model.default_cfg = _cfg()

    return model

@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model

@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), **kwargs)

    return model


@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)

    return model


@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models (
        img_size=img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial (nn.LayerNorm, eps=1e-6), **kwargs)

    return model


class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if len(self.reduction_ratio) == 4:
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif len(self.reduction_ratio) == 3:
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if len(self.reduction_ratio) == 4:
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if len(self.reduction_ratio) == 3:
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out

class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class LocalEnhance_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        x_atten = ((attn_score @ v) + local_pos).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == "mix":
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx

class FuseTransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, fuse_mode="add"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx

class MiT(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList(
            [TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList(
            [TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList(
            [TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class FuseMiT(nn.Module):
    def __init__(self, image_size, dims, layers, fuse_mode="add"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList(
            [FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList(
            [FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList(
            [FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

class Decoder(nn.Module):
    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.conv_seg = nn.Conv2d(128, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)

segformer_settings = {
    "B0": [[32, 64, 160, 256], [2, 2, 2, 2], 256],  # [channel dimensions, num encoder layers, embed dim]
    "B1": [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    "B2": [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    "B3": [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    "B4": [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    "B5": [[64, 128, 320, 512], [3, 6, 40, 3], 768],
}

class SegFormer(nn.Module):
    def __init__(self, model_name: str = "B0", num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert (
            model_name in segformer_settings.keys()
        ), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)