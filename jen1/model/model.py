from typing import Optional, Sequence

import torch
import torch.nn as nn
from einops import reduce
from torch import Tensor

from jen1.model.blocks import Patcher, DownsampleBlock1d, UpsampleBlock1d, BottleneckBlock1d, Unpatcher
from utils.module import rand_bool, TimePositionalEmbedding, FixedEmbedding, STFT
from utils.script_util import exists, default, groupby


class UNet1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int,
            multipliers: Sequence[int],
            factors: Sequence[int],
            num_blocks: Sequence[int],
            attentions: Sequence[int],
            patch_size: int = 1,
            resnet_groups: int = 8,
            use_context_time: bool = True,
            kernel_multiplier_downsample: int = 2,
            use_nearest_upsample: bool = False,
            use_skip_scale: bool = True,
            use_snake: bool = False,
            use_stft: bool = False,
            use_stft_context: bool = False,
            out_channels: Optional[int] = None,
            context_features: Optional[int] = None,
            context_features_multiplier: int = 4,
            context_channels: Optional[Sequence[int]] = None,
            context_embedding_features: Optional[int] = None,
            **kwargs,
    ):
        super().__init__()
        out_channels = default(out_channels, in_channels)
        context_channels = list(default(context_channels, []))
        num_layers = len(multipliers) - 1
        use_context_features = exists(context_features)
        use_context_channels = len(context_channels) > 0
        context_mapping_features = None

        attention_kwargs, kwargs = groupby("attention_", kwargs, keep_prefix=True)

        self.num_layers = num_layers
        self.use_context_time = use_context_time
        self.use_context_features = use_context_features
        self.use_context_channels = use_context_channels
        self.use_stft = use_stft
        self.use_stft_context = use_stft_context

        self.context_features = context_features
        context_channels_pad_length = num_layers + 1 - len(context_channels)
        context_channels = context_channels + [0] * context_channels_pad_length
        self.context_channels = context_channels
        self.context_embedding_features = context_embedding_features

        if use_context_channels:
            has_context = [c > 0 for c in context_channels]
            self.has_context = has_context
            self.channels_ids = [sum(has_context[:i]) for i in range(len(has_context))]

        assert (
                len(factors) == num_layers
                and len(attentions) >= num_layers
                and len(num_blocks) == num_layers
        )

        if use_context_time or use_context_features:
            context_mapping_features = channels * context_features_multiplier

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_stft:
            stft_kwargs, kwargs = groupby("stft_", kwargs)
            assert "num_fft" in stft_kwargs, "stft_num_fft required if use_stft=True"
            stft_channels = (stft_kwargs["num_fft"] // 2 + 1) * 2
            in_channels *= stft_channels
            out_channels *= stft_channels
            context_channels[0] *= stft_channels if use_stft_context else 1
            assert exists(in_channels) and exists(out_channels)
            self.stft = STFT(**stft_kwargs)

        assert not kwargs, f"Unknown arguments: {', '.join(list(kwargs.keys()))}"

        self.to_in = Patcher(
            in_channels=in_channels + context_channels[0],
            out_channels=channels * multipliers[0],
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

        self.downsamples = nn.ModuleList(
            [
                DownsampleBlock1d(
                    in_channels=channels * multipliers[i],
                    out_channels=channels * multipliers[i + 1],
                    context_mapping_features=context_mapping_features,
                    context_channels=context_channels[i + 1],
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i],
                    factor=factors[i],
                    kernel_multiplier=kernel_multiplier_downsample,
                    num_groups=resnet_groups,
                    use_pre_downsample=True,
                    use_skip=True,
                    use_snake=use_snake,
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in range(num_layers)
            ]
        )

        self.bottleneck = BottleneckBlock1d(
            channels=channels * multipliers[-1],
            context_mapping_features=context_mapping_features,
            context_embedding_features=context_embedding_features,
            num_groups=resnet_groups,
            num_transformer_blocks=attentions[-1],
            use_snake=use_snake,
            **attention_kwargs,
        )

        self.upsamples = nn.ModuleList(
            [
                UpsampleBlock1d(
                    in_channels=channels * multipliers[i + 1],
                    out_channels=channels * multipliers[i],
                    context_mapping_features=context_mapping_features,
                    context_embedding_features=context_embedding_features,
                    num_layers=num_blocks[i] + (1 if attentions[i] else 0),
                    factor=factors[i],
                    use_nearest=use_nearest_upsample,
                    num_groups=resnet_groups,
                    use_skip_scale=use_skip_scale,
                    use_pre_upsample=False,
                    use_skip=True,
                    use_snake=use_snake,
                    skip_channels=channels * multipliers[i + 1],
                    num_transformer_blocks=attentions[i],
                    **attention_kwargs,
                )
                for i in reversed(range(num_layers))
            ]
        )

        self.to_out = Unpatcher(
            in_channels=channels * multipliers[0],
            out_channels=out_channels,
            patch_size=patch_size,
            context_mapping_features=context_mapping_features,
            use_snake=use_snake
        )

    def get_channels(
            self, channels_list: Optional[Sequence[Tensor]] = None, layer: int = 0
    ):
        use_context_channels = self.use_context_channels and self.has_context[layer]
        if not use_context_channels:
            return None
        assert exists(channels_list), 'Missing context'
        # Get channels index (skipping zero channel contexts)
        channels_id = self.channels_ids[layer]
        # Get channels
        channels = channels_list[channels_id]
        message = f"Missing context for layer {layer} at index {channels_id}"
        assert exists(channels), message
        # Check channels
        num_channels = self.context_channels[layer]
        message = f"Expected context with {num_channels} channels at idx {channels_id}"
        assert channels.shape[1] == num_channels, message
        # STFT channels if requested
        channels = self.stft.encode1d(channels) if self.use_stft_context else channels  # type: ignore # noqa
        return channels

    def get_mapping(
            self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]
        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)
        return mapping

    def forward(
            self,
            x: Tensor,
            time: Optional[Tensor] = None,
            *,
            features: Optional[Tensor] = None,
            channels_list: Optional[Sequence[Tensor]] = None,
            embedding: Optional[Tensor] = None,
            embedding_mask: Optional[Tensor] = None,
            causal: Optional[bool] = False,
    ):
        channels = self.get_channels(channels_list, layer=0)
        # Apply stft if required
        x = self.stft.encode1d(x) if self.use_stft else x  # type: ignore
        # Concat context channels at layer 0 if provided
        x = torch.cat([x, channels], dim=1) if exists(channels) else x
        # Compute mapping from time and features
        mapping = self.get_mapping(time, features)
        x = self.to_in(x, mapping)
        skips_list = [x]

        for i, downsample in enumerate(self.downsamples):
            channels = self.get_channels(channels_list, layer=i + 1)
            x, skips = downsample(
                x, mapping=mapping, channels=channels, embedding=embedding,
                embedding_mask=embedding_mask, causal=causal
            )
            skips_list += [skips]
        x = self.bottleneck(x, mapping=mapping, embedding=embedding,
                            embedding_mask=embedding_mask, causal=causal)

        for i, upsample in enumerate(self.upsamples):
            skips = skips_list.pop()
            x = upsample(x, skips=skips, mapping=mapping, embedding=embedding,
                         embedding_mask=embedding_mask, causal=causal)

        x += skips_list.pop()
        x = self.to_out(x, mapping)
        x = self.stft.decoded1d(x) if self.use_stft else x

        return x


class UNetCFG1d(UNet1d):
    """UNet1d with Classifier-Free Guidance"""

    def __init__(
            self,
            context_embedding_max_length: int,
            context_embedding_features: int,
            use_xattn_time: bool = False,
            **kwargs,
    ):
        super().__init__(
            context_embedding_features=context_embedding_features, **kwargs
        )

        self.use_xattn_time = use_xattn_time

        if use_xattn_time:
            assert exists(context_embedding_features)
            self.to_time_embedding = nn.Sequential(
                TimePositionalEmbedding(
                    dim=kwargs["channels"], out_features=context_embedding_features
                ),
                nn.GELU(),
            )

            context_embedding_max_length += 1  # Add one for time embedding

        self.fixed_embedding = FixedEmbedding(
            max_length=context_embedding_max_length, features=context_embedding_features
        )

    def forward(  # type: ignore
            self,
            x: Tensor,
            time: Tensor,
            *,
            embedding: Tensor,
            embedding_mask: Optional[Tensor] = None,
            embedding_scale: float = 1.0,
            embedding_mask_proba: float = 0.0,
            batch_cfg: bool = False,
            scale_cfg: bool = False,
            scale_phi: float = 0.7,
            **kwargs,
    ) -> Tensor:
        b, device = embedding.shape[0], embedding.device

        if self.use_xattn_time:
            embedding = torch.cat([embedding, self.to_time_embedding(time).unsqueeze(1)], dim=1)

            if embedding_mask is not None:
                embedding_mask = torch.cat([embedding_mask, torch.ones((b, 1), device=device)], dim=1)

        fixed_embedding = self.fixed_embedding(embedding)

        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            if batch_cfg:
                batch_x = torch.cat([x, x], dim=0)
                batch_time = torch.cat([time, time], dim=0)
                batch_embed = torch.cat([embedding, fixed_embedding], dim=0)
                batch_mask = None
                if embedding_mask is not None:
                    batch_mask = torch.cat([embedding_mask, embedding_mask], dim=0)

                batch_features = None
                features = kwargs.pop("features", None)
                if self.use_context_features:
                    batch_features = torch.cat([features, features], dim=0)

                batch_channels = None
                channels_list = kwargs.pop("channels_list", None)
                if self.use_context_channels:
                    batch_channels = []
                    for channels in channels_list:
                        batch_channels += [torch.cat([channels, channels], dim=0)]

                # Compute both normal and fixed embedding outputs
                batch_out = super().forward(batch_x, batch_time, embedding=batch_embed, embedding_mask=batch_mask,
                                            features=batch_features, channels_list=batch_channels, **kwargs)
                out, out_masked = batch_out.chunk(2, dim=0)

            else:
                # Compute both normal and fixed embedding outputs
                out = super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)
                out_masked = super().forward(x, time, embedding=fixed_embedding, embedding_mask=embedding_mask,
                                             **kwargs)

            out_cfg = out_masked + (out - out_masked) * embedding_scale

            if scale_cfg:

                out_std = out.std(dim=1, keepdim=True)
                out_cfg_std = out_cfg.std(dim=1, keepdim=True)

                return scale_phi * (out_cfg * (out_std / out_cfg_std)) + (1 - scale_phi) * out_cfg

            else:

                return out_cfg

        else:
            return super().forward(x, time, embedding=embedding, embedding_mask=embedding_mask, **kwargs)
