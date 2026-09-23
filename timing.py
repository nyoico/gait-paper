"""TIMING attribution adapted to the two-sensor sequence classifier in model.py.

This is an independent implementation of the segment-masked Integrated
Gradients procedure described in the TIMING paper. The explainer expects a
callable that maps a batch shaped [B, T, D] to class scores/probabilities
shaped [B, C].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn


@dataclass
class TimingResult:
    """Outputs from TIMING attribution."""

    attribution: Tensor
    interpolated_count: Tensor
    targets: Tensor
    original_output: Tensor


class GRFClassifierWrapper(nn.Module):
    """Expose the current two-input model as a [B, T, 2D] -> [B, C] model.

    The first ``sensor_dim`` features are interpreted as the left sensor and
    the remaining features as the right sensor. The wrapped model still uses
    the same one-token decoder input used by train.py.
    """

    def __init__(
        self,
        base_model: nn.Module,
        sensor_dim: int,
        return_probabilities: bool = True,
        class_groups: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.sensor_dim = int(sensor_dim)
        self.return_probabilities = bool(return_probabilities)
        self.class_groups = (
            None
            if class_groups is None
            else tuple(tuple(int(index) for index in group) for group in class_groups)
        )
        if self.class_groups is not None and not self.return_probabilities:
            raise ValueError("Grouped outputs require return_probabilities=True")

    def forward(self, x_btd: Tensor) -> Tensor:
        if x_btd.ndim != 3:
            raise ValueError(f"Expected [B, T, 2D], got {tuple(x_btd.shape)}")
        if x_btd.size(-1) != 2 * self.sensor_dim:
            raise ValueError(
                "Last dimension must equal 2 * sensor_dim: "
                f"got {x_btd.size(-1)} and sensor_dim={self.sensor_dim}."
            )

        left_bct = x_btd[..., : self.sensor_dim].transpose(1, 2).contiguous()
        right_bct = x_btd[..., self.sensor_dim :].transpose(1, 2).contiguous()

        batch_size = x_btd.size(0)
        tgt_seq = torch.zeros(
            (batch_size, 1),
            dtype=torch.long,
            device=x_btd.device,
        )

        # model.py returns [decoder_length, B, C]. train.py uses the final step.
        logits = self.base_model(left_bct, right_bct, tgt_seq)[-1]
        if not self.return_probabilities:
            return logits

        probabilities = torch.softmax(logits, dim=-1)
        if self.class_groups is None:
            return probabilities

        grouped = [probabilities[:, group].sum(dim=-1) for group in self.class_groups]
        return torch.stack(grouped, dim=-1)


def _expand_baseline(inputs: Tensor, baseline: Optional[Tensor]) -> Tensor:
    if baseline is None:
        return torch.zeros_like(inputs)

    baseline = baseline.to(device=inputs.device, dtype=inputs.dtype)
    try:
        return torch.broadcast_to(baseline, inputs.shape).clone()
    except RuntimeError as exc:
        raise ValueError(
            f"Baseline shape {tuple(baseline.shape)} cannot broadcast to "
            f"input shape {tuple(inputs.shape)}."
        ) from exc


def _segment_mask(
    *,
    alpha_count: int,
    batch_size: int,
    time_steps: int,
    feature_dim: int,
    num_segments: int,
    min_seg_len: int,
    max_seg_len: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """Create a mask where 1 means interpolated and 0 means retained input.

    For every integration step and every sample, ``num_segments`` contiguous
    temporal segments are selected. Each segment belongs to one randomly
    chosen feature. Segment overlap is allowed, as in the stochastic TIMING
    construction.
    """

    mask = torch.ones(
        (alpha_count, batch_size, time_steps, feature_dim),
        device=device,
        dtype=dtype,
    )
    if num_segments == 0:
        return mask

    dims = torch.randint(
        low=0,
        high=feature_dim,
        size=(alpha_count, batch_size, num_segments),
        device=device,
        generator=generator,
    )
    lengths = torch.randint(
        low=min_seg_len,
        high=max_seg_len + 1,
        size=(alpha_count, batch_size, num_segments),
        device=device,
        generator=generator,
    )

    # A segment of length L has T-L+1 valid starting positions, including the
    # final possible start. Generate starts with a continuous draw so the
    # upper limit may differ for each sampled length.
    start_scale = (time_steps - lengths + 1).to(dtype=torch.float32)
    starts = torch.floor(
        torch.rand(
            lengths.shape,
            device=device,
            generator=generator,
            dtype=torch.float32,
        )
        * start_scale
    ).long()

    for segment_idx in range(num_segments):
        segment_lengths = lengths[:, :, segment_idx]
        segment_starts = starts[:, :, segment_idx]
        segment_dims = dims[:, :, segment_idx]
        longest = int(segment_lengths.max().item())

        # This loop is over temporal offsets, while all alpha steps and batch
        # elements are indexed together.
        for offset in range(longest):
            valid = offset < segment_lengths
            if not bool(valid.any()):
                continue
            alpha_idx, batch_idx = valid.nonzero(as_tuple=True)
            time_idx = segment_starts[alpha_idx, batch_idx] + offset
            feature_idx = segment_dims[alpha_idx, batch_idx]
            mask[alpha_idx, batch_idx, time_idx, feature_idx] = 0.0

    return mask


def timing_attribute(
    forward_func: Callable[[Tensor], Tensor],
    inputs: Tensor,
    *,
    baseline: Optional[Tensor] = None,
    targets: Optional[Tensor] = None,
    n_steps: int = 50,
    num_segments: int = 20,
    min_seg_len: int = 1,
    max_seg_len: Optional[int] = None,
    alpha_batch_size: int = 5,
    seed: int = 42,
) -> TimingResult:
    """Compute TIMING attribution for a batch of multivariate sequences.

    Args:
        forward_func: Callable mapping [B, T, D] to [B, C]. For paper-aligned
            CPD/CPP analysis, return class probabilities rather than logits.
        inputs: Normalized model inputs with shape [B, T, D].
        baseline: Baseline broadcastable to ``inputs``. For standardized data,
            a zero baseline represents the channel-wise training mean.
        targets: Class index per sample. If omitted, the model prediction on
            the unmodified input is used.
        n_steps: Number of path integration samples. The sampled alphas are
            0, 1/n_steps, ..., (n_steps-1)/n_steps.
        num_segments: Number of retained contiguous segments per integration
            sample and per batch element.
        min_seg_len: Minimum segment length in time points.
        max_seg_len: Maximum segment length; defaults to the full sequence.
        alpha_batch_size: Number of integration samples evaluated in one model
            forward pass. Reduce this value if GPU memory is insufficient.
        seed: Random-mask seed.

    Returns:
        ``TimingResult`` with signed attribution shaped [B, T, D].
    """

    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [B, T, D], got {inputs.shape}")
    if not inputs.is_floating_point():
        raise TypeError("inputs must be floating point so gradients can be taken")

    batch_size, time_steps, feature_dim = inputs.shape
    if batch_size < 1 or time_steps < 1 or feature_dim < 1:
        raise ValueError("All input dimensions must be non-zero")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if alpha_batch_size < 1:
        raise ValueError("alpha_batch_size must be at least 1")
    if num_segments < 0:
        raise ValueError("num_segments cannot be negative")

    max_seg_len = time_steps if max_seg_len is None else int(max_seg_len)
    min_seg_len = int(min_seg_len)
    if not 1 <= min_seg_len <= max_seg_len <= time_steps:
        raise ValueError(
            "Segment lengths must satisfy "
            f"1 <= min_seg_len <= max_seg_len <= T; got "
            f"{min_seg_len}, {max_seg_len}, T={time_steps}."
        )

    baseline_full = _expand_baseline(inputs, baseline)
    detached_inputs = inputs.detach()
    detached_baseline = baseline_full.detach()

    with torch.no_grad():
        original_output = forward_func(detached_inputs)
        if original_output.ndim != 2 or original_output.size(0) != batch_size:
            raise ValueError(
                "forward_func must return [B, C], got "
                f"{tuple(original_output.shape)}"
            )
        if targets is None:
            targets = original_output.argmax(dim=-1)
        else:
            targets = targets.to(device=inputs.device, dtype=torch.long)
            if targets.shape != (batch_size,):
                raise ValueError(
                    f"targets must have shape [{batch_size}], got {targets.shape}"
                )

    if torch.any(targets < 0) or torch.any(targets >= original_output.size(1)):
        raise ValueError("At least one target class index is out of range")

    total_gradient = torch.zeros_like(detached_inputs)
    interpolated_count = torch.zeros_like(detached_inputs)
    input_delta = detached_inputs - detached_baseline
    alphas = torch.arange(
        n_steps,
        device=inputs.device,
        dtype=inputs.dtype,
    ) / float(n_steps)

    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(int(seed))

    for alpha_start in range(0, n_steps, alpha_batch_size):
        alpha_chunk = alphas[alpha_start : alpha_start + alpha_batch_size]
        alpha_count = alpha_chunk.numel()

        # [A, B, T, D]
        interpolated = (
            detached_baseline.unsqueeze(0)
            + alpha_chunk.view(alpha_count, 1, 1, 1)
            * input_delta.unsqueeze(0)
        )
        time_mask = _segment_mask(
            alpha_count=alpha_count,
            batch_size=batch_size,
            time_steps=time_steps,
            feature_dim=feature_dim,
            num_segments=num_segments,
            min_seg_len=min_seg_len,
            max_seg_len=max_seg_len,
            device=inputs.device,
            dtype=inputs.dtype,
            generator=generator,
        )

        # Retained segments stay exactly at the original input; all other cells
        # lie on the baseline-to-input interpolation path.
        masked = (
            time_mask * interpolated
            + (1.0 - time_mask) * detached_inputs.unsqueeze(0)
        ).detach()
        masked.requires_grad_(True)

        flat_masked = masked.reshape(
            alpha_count * batch_size,
            time_steps,
            feature_dim,
        )

        with torch.enable_grad():
            output = forward_func(flat_masked)
            if output.ndim != 2 or output.size(0) != alpha_count * batch_size:
                raise ValueError(
                    "forward_func must return [A*B, C] for expanded inputs, got "
                    f"{tuple(output.shape)}"
                )
            expanded_targets = targets.unsqueeze(0).expand(alpha_count, -1).reshape(-1)
            selected = output.gather(1, expanded_targets.unsqueeze(1)).sum()
            gradients = torch.autograd.grad(
                selected,
                masked,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )[0]

        # Do not attribute cells that were retained at the original input for
        # this path sample.
        gradients = gradients * time_mask
        total_gradient += gradients.sum(dim=0)
        interpolated_count += time_mask.sum(dim=0)

    attribution = input_delta * total_gradient / interpolated_count.clamp_min(1.0)
    attribution = torch.where(
        interpolated_count > 0,
        attribution,
        torch.zeros_like(attribution),
    )

    return TimingResult(
        attribution=attribution.detach(),
        interpolated_count=interpolated_count.detach(),
        targets=targets.detach(),
        original_output=original_output.detach(),
    )


@torch.no_grad()
def cumulative_probability_change(
    forward_prob: Callable[[Tensor], Tensor],
    inputs: Tensor,
    attributions: Tensor,
    *,
    baseline: Optional[Tensor] = None,
    k: Optional[int] = None,
    remove_largest_first: bool = True,
) -> Tensor:
    """Calculate per-sample CPD- or CPP-style cumulative probability change.

    ``remove_largest_first=True`` yields CPD: cells with the largest absolute
    attribution are replaced first and a higher score is preferred.
    ``False`` yields CPP: cells with the smallest absolute attribution are
    replaced first and a lower score is preferred.
    """

    if inputs.shape != attributions.shape or inputs.ndim != 3:
        raise ValueError("inputs and attributions must share shape [B, T, D]")

    baseline_full = _expand_baseline(inputs, baseline)
    batch_size = inputs.size(0)
    flat_size = inputs[0].numel()
    k = flat_size if k is None else int(k)
    if not 1 <= k <= flat_size:
        raise ValueError(f"k must be in [1, {flat_size}], got {k}")

    order = attributions.abs().reshape(batch_size, -1).argsort(
        dim=1,
        descending=remove_largest_first,
    )[:, :k]

    current = inputs.detach().clone()
    current_flat = current.reshape(batch_size, -1)
    baseline_flat = baseline_full.reshape(batch_size, -1)
    batch_indices = torch.arange(batch_size, device=inputs.device)

    previous = forward_prob(current)
    cumulative = torch.zeros(batch_size, device=inputs.device, dtype=previous.dtype)

    for step in range(k):
        index = order[:, step]
        current_flat[batch_indices, index] = baseline_flat[batch_indices, index]
        updated = forward_prob(current)
        cumulative += (updated - previous).abs().sum(dim=-1)
        previous = updated

    return cumulative
