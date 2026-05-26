import diffusers.models.adapter
import torch
from typing import Optional
import einops


class ImageAdapter(torch.nn.Module):
    def __init__(
        self, in_channels: int = 3,
        channels: list = [320, 320, 640, 1280, 1280],
        is_downblocks: list = [False, True, True, True, False],
        num_res_blocks: int = 2, downscale_factor: int = 8,
        use_zero_convs: bool = False, zero_gate_coef: Optional[float] = None,
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor ** 2
        self.unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        self.body = torch.nn.ModuleList([
            diffusers.models.adapter.AdapterBlock(
                in_channels if i == 0 else channels[i - 1], channels[i],
                num_res_blocks, down=is_downblocks[i])
            for i in range(len(channels))
        ])
        self.gradient_checkpointing = gradient_checkpointing

        self.zero_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(channel, channel, 1)
            for channel in channels
        ]) if use_zero_convs else [None for _ in channels]
        for i in self.zero_convs:
            if i is not None:
                torch.nn.init.zeros_(i.weight)
                torch.nn.init.zeros_(i.bias)

        self.zero_gate_coef = zero_gate_coef
        self.zero_gates = torch.nn.Parameter(torch.zeros(len(channels))) \
            if zero_gate_coef else None

    def forward(self, x: torch.Tensor, return_features: bool = False):
        base_shape = x.shape[:-3]
        x = self.unshuffle(x.flatten(0, -4))
        features = []
        for i, (block, zero_conv) in enumerate(zip(self.body, self.zero_convs)):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False)
            else:
                x = block(x)

            x_out = x
            if zero_conv is not None:
                x_out = zero_conv(x_out)

            if self.zero_gates is not None:
                x_out = x_out * torch.tanh(
                    self.zero_gate_coef * self.zero_gates[i])

            features.append(x_out.view(*base_shape, *x_out.shape[1:]))
        return features if not return_features else features[-1]


def zero_module(module: torch.nn.Module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def select_middle_frame_indices(
    dense_t: int,
    target_t: int,
    device: torch.device,
    temporal_downsample_factor: int = 4,
    group_middle_index: int = 1,
) -> torch.Tensor:
    """Select one deterministic condition frame for each latent time step.

    Two layouts are supported:
    - dense_t == target_t * 4:
      [0,1,2,3] -> choose index 1, [4,5,6,7] -> choose index 5, ...
    - target_t == 1 + (dense_t - 1) // 4:
      keep frame 0 for the first latent, then choose offset 1 inside each following
      4-frame chunk: [1,2,3,4] -> choose 2, [5,6,7,8] -> choose 6, ...
    """
    if target_t <= 0:
        raise ValueError(f"target_t must be positive, got {target_t}")

    if dense_t < target_t:
        raise ValueError(
            f"dense_t ({dense_t}) must be >= target_t ({target_t})"
        )

    if dense_t == target_t:
        return torch.arange(dense_t, device=device, dtype=torch.long)

    factor = int(temporal_downsample_factor)
    offset = int(group_middle_index)

    if factor <= 0:
        raise ValueError(f"temporal_downsample_factor must be positive, got {factor}")

    if offset < 0 or offset >= factor:
        raise ValueError(
            f"group_middle_index must be in [0, {factor - 1}], got {offset}"
        )

    if dense_t == target_t * factor:
        return (
            torch.arange(target_t, device=device, dtype=torch.long) * factor + offset
        )

    expected_target_t = 1 + (dense_t - 1) // factor
    if target_t == expected_target_t:
        head = torch.zeros(1, device=device, dtype=torch.long)
        if target_t == 1:
            return head

        tail = (
            1
            + torch.arange(target_t - 1, device=device, dtype=torch.long) * factor
            + offset
        )
        tail = tail.clamp(max=dense_t - 1)
        return torch.cat([head, tail], dim=0)

    raise ValueError(
        f"Cannot build deterministic middle-frame indices for dense_t={dense_t}, "
        f"target_t={target_t}, factor={factor}. Expected either dense_t == "
        f"target_t * factor or target_t == 1 + (dense_t - 1) // factor."
    )


class CausalConv3d(torch.nn.Conv3d):
    """Causal 3D convolution on temporal axis.

    Input shape is [B, C, T, H, W].
    Temporal padding is left-only.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        kernel_t = self.kernel_size[0]
        dilation_t = self.dilation[0]
        left_t = dilation_t * (kernel_t - 1)

        self._causal_padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            left_t,
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, self._causal_padding)
        return super().forward(x)

class TemporalConditionImageAdapter(torch.nn.Module):
    """Fixed-frame causal condition adapter.

    - Fixed temporal frame selection.
    - OpenDWM-style PixelUnshuffle + AdapterBlock.
    - Causal temporal mix after each adapter stage.
    - Output count = condition_inject_layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        temporal_downsample_factor: int = 4,
        condition_inject_layers: int = 8,
    ):
        super().__init__()

        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels or 768)
        self.temporal_downsample_factor = int(temporal_downsample_factor)
        self.condition_inject_layers = int(condition_inject_layers)
        self.group_middle_index = 1
        self.spatial_downscale_factor = 8
        self.gradient_checkpointing = True

        if self.temporal_downsample_factor != 4:
            raise ValueError(
                f"TemporalConditionImageAdapter expects temporal_downsample_factor=4, "
                f"got {self.temporal_downsample_factor}"
            )

        if self.condition_inject_layers <= 0:
            raise ValueError(
                f"condition_inject_layers must be positive, "
                f"got {self.condition_inject_layers}"
            )

        self.adapter_channels = [
            self.hidden_channels for _ in range(self.condition_inject_layers)
        ]

        self.adapter_is_downblocks = [
            True
        ] + [
            False for _ in range(self.condition_inject_layers - 1)
        ]

        unshuffle_in_channels = int(in_channels) * (
            self.spatial_downscale_factor ** 2
        )

        self.unshuffle = torch.nn.PixelUnshuffle(self.spatial_downscale_factor)

        self.adapter_blocks = torch.nn.ModuleList()
        self.zero_convs = torch.nn.ModuleList()
        self.causal_mixers = torch.nn.ModuleList()
        self.out_projs = torch.nn.ModuleList()

        for stage_id, channel in enumerate(self.adapter_channels):
            if stage_id == 0:
                block_in_channels = unshuffle_in_channels
            else:
                block_in_channels = self.adapter_channels[stage_id - 1]

            self.adapter_blocks.append(
                diffusers.models.adapter.AdapterBlock(
                    block_in_channels,
                    channel,
                    2,
                    down=self.adapter_is_downblocks[stage_id],
                )
            )

            zero_conv = torch.nn.Conv2d(
                in_channels=channel,
                out_channels=channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            torch.nn.init.zeros_(zero_conv.weight)
            torch.nn.init.zeros_(zero_conv.bias)
            self.zero_convs.append(zero_conv)

            self.causal_mixers.append(
                torch.nn.Sequential(
                    CausalConv3d(
                        in_channels=channel,
                        out_channels=channel,
                        kernel_size=(3, 1, 1),
                        stride=1,
                        padding=(1, 0, 0),
                        bias=True,
                    ),
                    torch.nn.SiLU(),
                    zero_module(
                        torch.nn.Conv3d(
                            in_channels=channel,
                            out_channels=channel,
                            kernel_size=(1, 1, 1),
                            stride=1,
                            padding=0,
                            bias=True,
                        )
                    ),
                )
            )

            self.out_projs.append(
                torch.nn.Conv3d(
                    in_channels=channel,
                    out_channels=self.out_channels,
                    kernel_size=(1, 1, 1),
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )

    def _select_temporal_indices(
        self,
        dense_t: int,
        target_t: int,
        device: torch.device,
    ) -> torch.Tensor:
        return select_middle_frame_indices(
            dense_t=dense_t,
            target_t=target_t,
            device=device,
            temporal_downsample_factor=self.temporal_downsample_factor,
            group_middle_index=self.group_middle_index,
        )

    def forward(
        self,
        condition_image_tensor: torch.Tensor,
        target_sequence_length: int,
        target_patch_size,
    ):
        if condition_image_tensor.ndim != 5:
            raise ValueError(
                f"condition_image_tensor must be 5D [(B*V), C, T, H, W], "
                f"but got shape {tuple(condition_image_tensor.shape)}"
            )

        if len(target_patch_size) != 2:
            raise ValueError(
                f"target_patch_size must be (patch_height, patch_width), "
                f"but got {target_patch_size}"
            )

        batch_size_total, _, dense_t, height, width = condition_image_tensor.shape
        patch_height, patch_width = target_patch_size

        if (
            height % self.spatial_downscale_factor != 0
            or width % self.spatial_downscale_factor != 0
        ):
            raise ValueError(
                "Condition image spatial size must be divisible by "
                f"{self.spatial_downscale_factor}, but got {(height, width)}."
            )

        x = condition_image_tensor.float()

        if bool((x.detach().amax() > 1.0).item()):
            x = x / 255.0

        temporal_indices = self._select_temporal_indices(
            dense_t=dense_t,
            target_t=target_sequence_length,
            device=x.device,
        )

        x = x.index_select(2, temporal_indices)

        x = einops.rearrange(
            x,
            "b c t h w -> (b t) c h w",
        )

        x = self.unshuffle(x)

        outputs = []

        for stage_id, block in enumerate(self.adapter_blocks):
            if self.training and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

            x_out = self.zero_convs[stage_id](x)

            if x_out.shape[-2:] != (patch_height, patch_width):
                if (
                    x_out.shape[-2] >= patch_height
                    and x_out.shape[-1] >= patch_width
                ):
                    x_aligned = torch.nn.functional.adaptive_max_pool2d(
                        x_out,
                        output_size=(patch_height, patch_width),
                    )
                else:
                    x_aligned = torch.nn.functional.interpolate(
                        x_out,
                        size=(patch_height, patch_width),
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                x_aligned = x_out

            x_3d = einops.rearrange(
                x_aligned,
                "(b t) c h w -> b c t h w",
                b=batch_size_total,
                t=target_sequence_length,
            )

            x_3d = x_3d + self.causal_mixers[stage_id](x_3d)

            y = self.out_projs[stage_id](x_3d)

            y = einops.rearrange(
                y,
                "b c t h w -> b (t h w) c",
                t=target_sequence_length,
                h=patch_height,
                w=patch_width,
            )

            outputs.append(y)

        return outputs