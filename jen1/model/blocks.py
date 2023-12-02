from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from dac.nn.layers import Snake1d
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from packaging import version
from torch import Tensor, einsum
from torch.backends.cuda import sdp_kernel
from torch.nn import functional as F

from utils.script_util import exists, default
from utils.module import crop

'''
def Conv1d(*args, **kwargs) -> nn.Module:
    return nn.Conv1d(*args, **kwargs)
'''

def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return nn.ConvTranspose1d(*args, **kwargs)

def Identity():
    class _Identity(nn.Module):
        def __init__(self):
            super(_Identity, self).__init__()
        def forward(self, x, causal=False):
            return x
    
    return _Identity()

def Conv1d(*args, **kwargs):
    class _Conv1d(nn.Module):
        def __init__(self):
            super(_Conv1d, self).__init__()
            self.kernel_size = kwargs.get('kernel_size', 1)
            self.dilation = kwargs.get('dilation', 1)
            self.padding =  kwargs.get('padding', 0)
            kwargs['padding'] = 0
            self.conv = nn.Conv1d(*args, **kwargs)

        def forward(self, x, causal):
            padding = (self.kernel_size - 1) * self.dilation
            if causal:
                x = F.pad(x, (padding, 0))
            else:
                half_padding = padding // 2
                x = F.pad(x, (half_padding, half_padding))
            return self.conv(x)

    return _Conv1d()

def Downsample1d(
        in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor,
        padding=factor * (kernel_multiplier // 2),
    )


def Upsample1d(
        in_channels: int, out_channels: int, factor: int, use_nearest: bool = False
) -> nn.Module:
    if factor == 1:
        return nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest"),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )
    else:
        return nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
        )


class ConvBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            num_groups: int = 8,
            use_norm: bool = True,
            use_snake: bool = False
    ) -> None:
        super().__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groupnorm = (
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            if use_norm
            else nn.Identity()
        )

        if use_snake:
            self.activation = Snake1d(in_channels)
        else:
            self.activation = nn.SiLU()

        self.project = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(
            self, x: Tensor, scale_shift: Optional[Tuple[Tensor, Tensor]] = None, causal=False
    ) -> Tensor:
        x = self.groupnorm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.activation(x)
        return self.project(x, causal)


class MappingToScaleShift(nn.Module):
    def __init__(
            self,
            features: int,
            channels: int,
    ):
        super().__init__()

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features=features, out_features=channels * 2),
        )

    def forward(self, mapping: Tensor) -> Tuple[Tensor, Tensor]:
        scale_shift = self.to_scale_shift(mapping)
        scale_shift = rearrange(scale_shift, "b c -> b c 1")
        scale, shift = scale_shift.chunk(2, dim=1)
        return scale, shift


class ResnetBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            use_norm: bool = True,
            use_snake: bool = False,
            num_groups: int = 8,
            context_mapping_features: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_mapping = exists(context_mapping_features)

        self.block1 = ConvBlock1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            use_norm=use_norm,
            num_groups=num_groups,
            use_snake=use_snake
        )

        if self.use_mapping:
            assert exists(context_mapping_features)
            self.to_scale_shift = MappingToScaleShift(
                features=context_mapping_features, channels=out_channels
            )

        self.block2 = ConvBlock1d(
            in_channels=out_channels,
            out_channels=out_channels,
            use_norm=use_norm,
            num_groups=num_groups,
            use_snake=use_snake
        )

        self.to_out = (
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else Identity()
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None, causal=False) -> Tensor:
        assert_message = "context mapping required if context_mapping_features > 0"
        assert not (self.use_mapping ^ exists(mapping)), assert_message

        h = self.block1(x, causal=causal)

        scale_shift = None
        if self.use_mapping:
            scale_shift = self.to_scale_shift(mapping)

        h = self.block2(h, scale_shift=scale_shift, causal=causal)

        return h + self.to_out(x, causal)


class Patcher(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            patch_size: int,
            context_mapping_features: Optional[int] = None,
            use_snake: bool = False,
    ):
        super().__init__()
        assert_message = f"out_channels must be divisible by patch_size ({patch_size})"
        assert out_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels,
            out_channels=out_channels // patch_size,
            num_groups=1,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None) -> Tensor:
        x = self.block(x, mapping)
        x = rearrange(x, "b c (l p) -> b (c p) l", p=self.patch_size)
        return x


class Unpatcher(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            patch_size: int,
            context_mapping_features: Optional[int] = None,
            use_snake: bool = False
    ):
        super().__init__()
        assert_message = f"in_channels must be divisible by patch_size ({patch_size})"
        assert in_channels % patch_size == 0, assert_message
        self.patch_size = patch_size

        self.block = ResnetBlock1d(
            in_channels=in_channels // patch_size,
            out_channels=out_channels,
            num_groups=1,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def forward(self, x: Tensor, mapping: Optional[Tensor] = None) -> Tensor:
        x = rearrange(x, " b (c p) l -> b c (l p) ", p=self.patch_size)
        x = self.block(x, mapping)
        return x


"""
Attention Components
"""


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim


def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask


class AttentionBase(nn.Module):
    def __init__(
            self,
            features: int,
            *,
            head_features: int,
            num_heads: int,
            out_features: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        out_features = default(out_features, features)

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=out_features
        )

        self.use_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        if not self.use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            # Use flash attention for A100 GPUs
            self.sdp_kernel_config = (True, False, False)
        else:
            # Don't use flash attention for other GPUs
            self.sdp_kernel_config = (False, True, True)

    def forward(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if exists(mask) else sim

            # Get attention matrix with softmax
            attn = sim.softmax(dim=-1, dtype=torch.float32)

            # Compute values
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            with sdp_kernel(*self.sdp_kernel_config):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
            self,
            features: int,
            *,
            head_features: int,
            num_heads: int,
            out_features: Optional[int] = None,
            context_features: Optional[int] = None,
            causal: bool = False,
    ):
        super().__init__()
        self.context_features = context_features
        self.causal = causal
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            out_features=out_features,
        )

    def forward(
            self,
            x: Tensor,  # [b, n, c]
            context: Optional[Tensor] = None,  # [b, m, d]
            context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
            causal: Optional[bool] = False,
    ) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask

        # Compute and return attention
        return self.attention(q, k, v, is_causal=self.causal or causal)


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
            self,
            features: int,
            num_heads: int,
            head_features: int,
            multiplier: int,
            context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None,
                causal: Optional[bool] = False) -> Tensor:
        x = self.attention(x, causal=causal) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context, context_mask=context_mask) + x
        x = self.feed_forward(x) + x
        return x


"""
Transformers
"""


class Transformer1d(nn.Module):
    def __init__(
            self,
            num_layers: int,
            channels: int,
            num_heads: int,
            head_features: int,
            multiplier: int,
            context_features: Optional[int] = None,
    ):
        super().__init__()
        
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.conv1d = Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.rearrange_in = Rearrange("b c t -> b t c")

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    context_features=context_features,
                )
                for i in range(num_layers)
            ]
        )
        
        self.rearrange_out =  Rearrange("b t c -> b c t")

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None, context_mask: Optional[Tensor] = None,
                causal=False) -> Tensor:
        x = self.group_norm(x)
        x = self.conv1d(x, causal)
        x = self.rearrange_in(x)
        for block in self.blocks:
            x = block(x, context=context, context_mask=context_mask, causal=causal)
        x = self.rearrange_out(x)
        x = self.conv1d(x, causal)
        return x


class DownsampleBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            factor: int,
            num_groups: int,
            num_layers: int,
            kernel_multiplier: int = 2,
            use_pre_downsample: bool = True,
            use_skip: bool = False,
            use_snake: bool = False,
            extract_channels: int = 0,
            context_channels: int = 0,
            num_transformer_blocks: int = 0,
            attention_heads: Optional[int] = None,
            attention_features: Optional[int] = None,
            attention_multiplier: Optional[int] = None,
            context_mapping_features: Optional[int] = None,
            context_embedding_features: Optional[int] = None,
    ):
        super().__init__()
        self.use_pre_downsample = use_pre_downsample
        self.use_skip = use_skip
        self.use_transformer = num_transformer_blocks > 0
        self.use_extract = extract_channels > 0
        self.use_context = context_channels > 0

        channels = out_channels if use_pre_downsample else in_channels

        self.downsample = Downsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            kernel_multiplier=kernel_multiplier,
        )

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + context_channels if i == 0 else channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                for i in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                    exists(attention_heads)
                    and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features
            )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def forward(
            self,
            x: Tensor,
            *,
            mapping: Optional[Tensor] = None,
            channels: Optional[Tensor] = None,
            embedding: Optional[Tensor] = None,
            embedding_mask: Optional[Tensor] = None,
            causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, List[Tensor]], Tensor]:

        if self.use_pre_downsample:
            x = self.downsample(x, causal)

        if self.use_context and exists(channels):
            x = torch.cat([x, channels], dim=1)

        skips = []
        for block in self.blocks:
            x = block(x, mapping=mapping, causal=causal)
            skips += [x] if self.use_skip else []

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
            skips += [x] if self.use_skip else []

        if not self.use_pre_downsample:
            x = self.downsample(x, causal)

        if self.use_extract:
            extracted = self.to_extracted(x, causal)
            return x, extracted

        return (x, skips) if self.use_skip else x


class UpsampleBlock1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            factor: int,
            num_layers: int,
            num_groups: int,
            use_nearest: bool = False,
            use_pre_upsample: bool = False,
            use_skip: bool = False,
            use_snake: bool = False,
            skip_channels: int = 0,
            use_skip_scale: bool = False,
            extract_channels: int = 0,
            num_transformer_blocks: int = 0,
            attention_heads: Optional[int] = None,
            attention_features: Optional[int] = None,
            attention_multiplier: Optional[int] = None,
            context_mapping_features: Optional[int] = None,
            context_embedding_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_extract = extract_channels > 0
        self.use_pre_upsample = use_pre_upsample
        self.use_transformer = num_transformer_blocks > 0
        self.use_skip = use_skip
        self.skip_scale = 2 ** -0.5 if use_skip_scale else 1.0

        channels = out_channels if use_pre_upsample else in_channels

        self.blocks = nn.ModuleList(
            [
                ResnetBlock1d(
                    in_channels=channels + skip_channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    context_mapping_features=context_mapping_features,
                    use_snake=use_snake
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_transformer:
            assert (
                    exists(attention_heads)
                    and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.upsample = Upsample1d(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=factor,
            use_nearest=use_nearest,
        )

        if self.use_extract:
            num_extract_groups = min(num_groups, extract_channels)
            self.to_extracted = ResnetBlock1d(
                in_channels=out_channels,
                out_channels=extract_channels,
                num_groups=num_extract_groups,
                use_snake=use_snake
            )

    def add_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        x, skip = crop(x, skip)
        return torch.cat([x, skip * self.skip_scale], dim=1)

    def forward(
            self,
            x: Tensor,
            *,
            skips: Optional[List[Tensor]] = None,
            mapping: Optional[Tensor] = None,
            embedding: Optional[Tensor] = None,
            embedding_mask: Optional[Tensor] = None,
            causal: Optional[bool] = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:

        if self.use_pre_upsample:
            x = self.upsample(x)

        for block in self.blocks:
            x = self.add_skip(x, skip=skips.pop()) if exists(skips) else x
            x = block(x, mapping=mapping, causal=causal)

        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)

        if not self.use_pre_upsample:
            x = self.upsample(x)

        if self.use_extract:
            extracted = self.to_extracted(x, causal=causal)
            return x, extracted

        return x


class BottleneckBlock1d(nn.Module):
    def __init__(
            self,
            channels: int,
            *,
            num_groups: int,
            num_transformer_blocks: int = 0,
            attention_heads: Optional[int] = None,
            attention_features: Optional[int] = None,
            attention_multiplier: Optional[int] = None,
            context_mapping_features: Optional[int] = None,
            context_embedding_features: Optional[int] = None,
            use_snake: bool = False,
    ):
        super().__init__()
        self.use_transformer = num_transformer_blocks > 0

        self.pre_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        if self.use_transformer:
            assert (
                    exists(attention_heads)
                    and exists(attention_multiplier)
            )

            attention_features = default(attention_features, channels // attention_heads)

            self.transformer = Transformer1d(
                num_layers=num_transformer_blocks,
                channels=channels,
                num_heads=attention_heads,
                head_features=attention_features,
                multiplier=attention_multiplier,
                context_features=context_embedding_features,
            )

        self.post_block = ResnetBlock1d(
            in_channels=channels,
            out_channels=channels,
            num_groups=num_groups,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def forward(
            self,
            x: Tensor,
            *,
            mapping: Optional[Tensor] = None,
            embedding: Optional[Tensor] = None,
            embedding_mask: Optional[Tensor] = None,
            causal: Optional[bool] = False
    ) -> Tensor:
        x = self.pre_block(x, mapping=mapping, causal=causal)
        if self.use_transformer:
            x = self.transformer(x, context=embedding, context_mask=embedding_mask, causal=causal)
        x = self.post_block(x, mapping=mapping, causal=causal)
        return x
