import torch.nn as nn

from typing import Tuple, Type, Optional, Dict, Any


class ResidualEncoder(nn.Module):
    def __init__(
        self,
        stem_cls: Type[nn.Module],
        block_cls: Type[nn.Module],
        num_in_channel: int,
        layers: Tuple[int, ...],
        channels: Tuple[int, ...],
        pool_shape: Tuple[int, int],
        block_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        assert len(layers) == len(channels), "`layers` & `channels` must align"

        self.stem = stem_cls(in_channel=num_in_channel, out_channel=channels[0])

        in_channel = channels[0]
        stages = []
        for n_blocks, out_channel in zip(layers, channels):
            stride = 1 if in_channel == out_channel else (2, 2)
            stages.append(
                self._make_stage(
                    block_cls,
                    in_channel,
                    out_channel,
                    n_blocks,
                    stride,
                    extra_kw=block_kwargs,
                )
            )
            in_channel = out_channel
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(pool_shape)
        self.out_channels = in_channel
        self.out_pool = pool_shape

    @staticmethod
    def _make_stage(block_cls, in_channel, out_channel, n_blocks, stride, extra_kw):
        layers = [
            block_cls(
                in_channel,
                out_channel,
                stride,
                **extra_kw,
            )
        ]
        for _ in range(1, n_blocks):
            layers.append(
                block_cls(
                    out_channel,
                    out_channel,
                    (1, 1),
                    **extra_kw,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.pool(x)
