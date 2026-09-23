import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_helper import parse_args


class _LearnedFakeQuantizeFn(torch.autograd.Function):
    """Uniform quantize-dequantize with STE and a learnable step size."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        scale_safe = scale.abs().clamp_min(1e-8)
        x_scaled = x / scale_safe
        x_clamped = x_scaled.clamp(qmin, qmax)
        x_rounded = torch.round(x_clamped)
        y = x_rounded * scale_safe

        ctx.save_for_backward(x_scaled, x_rounded, scale_safe)
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.numel = x.numel()
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_scaled, x_rounded, scale_safe = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax

        # STE for x. Gradients outside the clipping range are suppressed.
        inside = (x_scaled >= qmin) & (x_scaled <= qmax)
        grad_x = grad_output * inside.to(grad_output.dtype)

        # LSQ-style scale gradient. This makes the quantization interval trainable.
        grad_scale_element = torch.where(
            x_scaled < qmin,
            torch.full_like(x_scaled, float(qmin)),
            torch.where(
                x_scaled > qmax,
                torch.full_like(x_scaled, float(qmax)),
                x_rounded - x_scaled,
            ),
        )
        grad_factor = 1.0 / math.sqrt(max(ctx.numel * max(abs(qmin), abs(qmax)), 1))
        grad_scale = (grad_output * grad_scale_element).sum().reshape_as(scale_safe)
        grad_scale = grad_scale * grad_factor

        return grad_x, grad_scale, None, None
    
    @staticmethod
    @parse_args("v", "v", "i", "i")
    def symbolic(g, x, scale, qmin, qmax):
        """
        Convert the QAT fake quantizer to ONNX QDQ operations.

        The existing implementation uses symmetric quantization, so zero_point is always 0.
        """

        if qmin < 0:
            zero_point_value = torch.tensor(
                0,
                dtype=torch.int8,
            )
        else:
            zero_point_value = torch.tensor(
                0,
                dtype=torch.uint8,
            )

        zero_point = g.op(
            "Constant",
            value_t=zero_point_value,
        )

        quantized = g.op(
            "QuantizeLinear",
            x,
            scale,
            zero_point,
        )

        dequantized = g.op(
            "DequantizeLinear",
            quantized,
            scale,
            zero_point,
        )

        return dequantized.setType(x.type())


class LearnedStepFakeQuantizer(nn.Module):
    """
    Per-tensor learned-step fake quantizer.

    During QAT it returns a floating-point tensor whose values are restricted to
    quantization levels. It does not itself provide an integer runtime kernel.
    """

    def __init__(self, bits: int = 8, signed: bool = True, name: str = ""):
        super().__init__()
        if bits < 2:
            raise ValueError(f"bits must be >= 2, got {bits}")

        self.bits = int(bits)
        self.signed = bool(signed)
        self.name = name

        if self.signed:
            self.qmin = -(2 ** (self.bits - 1))
            self.qmax = 2 ** (self.bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**self.bits - 1

        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.enabled = False

    @torch.no_grad()
    def initialize_from(self, x: torch.Tensor) -> None:
        x_detached = x.detach()
        if self.signed:
            max_abs = x_detached.abs().max()
            initial_scale = max_abs / max(self.qmax, 1)
        else:
            max_value = x_detached.max().clamp_min(0.0)
            initial_scale = max_value / max(self.qmax, 1)

        self.scale.copy_(initial_scale.clamp_min(1e-8))
        self.initialized.fill_(True)

    @torch.no_grad()
    def reset(self) -> None:
        self.scale.fill_(1.0)
        self.initialized.fill_(False)

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        if not bool(self.initialized.item()):
            self.initialize_from(x)

        return _LearnedFakeQuantizeFn.apply(
            x,
            self.scale,
            self.qmin,
            self.qmax,
        )

    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, signed={self.signed}, enabled={self.enabled}, "
            f"initialized={bool(self.initialized.item())}"
        )


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8,
        activation_signed: bool = True,
        quantize_output: bool = False,
        output_signed: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_quant = LearnedStepFakeQuantizer(
            bits=weight_bits,
            signed=True,
            name="weight",
        )
        self.activation_quant = LearnedStepFakeQuantizer(
            bits=activation_bits,
            signed=activation_signed,
            name="activation",
        )
        self.output_quant: Optional[LearnedStepFakeQuantizer]
        if quantize_output:
            self.output_quant = LearnedStepFakeQuantizer(
                bits=activation_bits,
                signed=output_signed,
                name="output_activation",
            )
        else:
            self.output_quant = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.activation_quant(x)
        weight_q = self.weight_quant(self.weight)
        y = F.linear(x_q, weight_q, self.bias)
        if self.output_quant is not None:
            y = self.output_quant(y)
        return y


class QuantConv1d(nn.Conv1d):
    def __init__(
        self,
        *args,
        weight_bits: int = 8,
        activation_bits: int = 8,
        activation_signed: bool = True,
        quantize_output: bool = False,
        output_signed: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight_quant = LearnedStepFakeQuantizer(
            bits=weight_bits,
            signed=True,
            name="weight",
        )
        self.activation_quant = LearnedStepFakeQuantizer(
            bits=activation_bits,
            signed=activation_signed,
            name="activation",
        )
        self.output_quant: Optional[LearnedStepFakeQuantizer]
        if quantize_output:
            self.output_quant = LearnedStepFakeQuantizer(
                bits=activation_bits,
                signed=output_signed,
                name="output_activation",
            )
        else:
            self.output_quant = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.activation_quant(x)
        weight_q = self.weight_quant(self.weight)
        y = F.conv1d(
            x_q,
            weight_q,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.output_quant is not None:
            y = self.output_quant(y)
        return y


class PosEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for this sinusoidal encoding")

        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class QuantMultiheadAttention(nn.Module):
    """Explicit multi-head attention so Q/K/V/out projections can be quantized."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        weight_bits: int,
        activation_bits: int,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        projection_kwargs = dict(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=True,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=True,
            quantize_output=True,
            output_signed=True,
        )
        self.q_proj = QuantLinear(**projection_kwargs)
        self.k_proj = QuantLinear(**projection_kwargs)
        self.v_proj = QuantLinear(**projection_kwargs)
        self.out_proj = QuantLinear(
            embed_dim,
            embed_dim,
            bias=True,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=True,
            quantize_output=False,
        )
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [L, B, E] -> [B, H, L, D]
        length, batch, _ = x.shape
        return (
            x.permute(1, 0, 2)
            .reshape(batch, length, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, L, D] -> [L, B, E]
        batch, _, length, _ = x.shape
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch, length, self.embed_dim)
            .permute(1, 0, 2)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(
                    attn_mask.unsqueeze(0).unsqueeze(0),
                    torch.finfo(scores.dtype).min,
                )
            else:
                scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )

        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)
        context = torch.matmul(attention, v)
        output = self.out_proj(self._merge_heads(context))

        weights = attention.mean(dim=1) if need_weights else None
        return output, weights


class QuantConvEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        weight_bits: int,
        activation_bits: int,
    ):
        super().__init__()
        self.conv = QuantConv1d(
            in_channels,
            embed_dim,
            kernel_size=1,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=True,
            quantize_output=False,
        )
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class QuantEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        weight_bits: int,
        activation_bits: int,
    ):
        super().__init__()
        self.self_attn = QuantMultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            weight_bits,
            activation_bits,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff1 = QuantConv1d(
            embed_dim,
            ff_dim,
            kernel_size=1,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=True,
        )
        self.ff2 = QuantConv1d(
            ff_dim,
            embed_dim,
            kernel_size=1,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=False,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(src, src, src, need_weights=False)
        src = self.norm1(src + self.dropout(attn_out))

        y = self.ff1(src.permute(1, 2, 0))
        y = F.relu(y)
        y = self.ff2(y).permute(2, 0, 1)
        src = self.norm2(src + self.dropout(y))
        return src


class QuantDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        weight_bits: int,
        activation_bits: int,
    ):
        super().__init__()
        self.masked_attn = QuantMultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            weight_bits,
            activation_bits,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = QuantMultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            weight_bits,
            activation_bits,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff1 = QuantConv1d(
            embed_dim,
            ff_dim,
            kernel_size=1,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=True,
        )
        self.ff2 = QuantConv1d(
            ff_dim,
            embed_dim,
            kernel_size=1,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_signed=False,
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_pad_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m1, _ = self.masked_attn(
            tgt,
            tgt,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_pad_mask,
            need_weights=False,
        )
        tgt = self.norm1(tgt + self.dropout(m1))

        m2, _ = self.cross_attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout(m2))

        y = self.ff1(tgt.permute(1, 2, 0))
        y = F.relu(y)
        y = self.ff2(y).permute(2, 0, 1)
        tgt = self.norm3(tgt + self.dropout(y))
        return tgt


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )


def make_casual_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # Compatibility with the misspelled helper in the original project.
    return make_causal_mask(seq_len, device)


class QKDStudentModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        sensor_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_len: int,
        weight_bits: int = 8,
        activation_bits: int = 8,
        first_last_bits: int = 8,
    ):
        super().__init__()
        self.config: Dict[str, Any] = {
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "sensor_dim": sensor_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "dropout": dropout,
            "max_len": max_len,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "first_last_bits": first_last_bits,
        }

        self.sensorL_embed = QuantConvEmbedding(
            sensor_dim,
            embed_dim,
            weight_bits=first_last_bits,
            activation_bits=first_last_bits,
        )
        self.sensorR_embed = QuantConvEmbedding(
            sensor_dim,
            embed_dim,
            weight_bits=first_last_bits,
            activation_bits=first_last_bits,
        )
        self.pos_enc = PosEncoding(embed_dim, max_len=max_len, dropout=dropout)

        self.encoder = nn.ModuleList(
            [
                QuantEncoderLayer(
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout,
                    weight_bits,
                    activation_bits,
                )
                for _ in range(num_layers)
            ]
        )

        self.tgt_embed = nn.Embedding(num_classes, embed_dim)
        self.decoder = nn.ModuleList(
            [
                QuantDecoderLayer(
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout,
                    weight_bits,
                    activation_bits,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = QuantLinear(
            embed_dim,
            num_classes,
            bias=True,
            weight_bits=first_last_bits,
            activation_bits=first_last_bits,
            activation_signed=True,
            quantize_output=False,
        )

    def quantizers(self):
        for module in self.modules():
            if isinstance(module, LearnedStepFakeQuantizer):
                yield module

    def enable_quantization(self) -> None:
        for quantizer in self.quantizers():
            quantizer.enable()

    def disable_quantization(self) -> None:
        for quantizer in self.quantizers():
            quantizer.disable()

    def reset_quantizers(self) -> None:
        for quantizer in self.quantizers():
            quantizer.reset()

    def quantizer_summary(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for name, module in self.named_modules():
            if isinstance(module, LearnedStepFakeQuantizer):
                summary[name] = {
                    "bits": module.bits,
                    "signed": module.signed,
                    "enabled": module.enabled,
                    "initialized": bool(module.initialized.item()),
                    "scale": float(module.scale.detach().cpu().item()),
                    "qmin": module.qmin,
                    "qmax": module.qmax,
                }
        return summary

    def forward(
        self,
        sensorL: torch.Tensor,
        sensorR: torch.Tensor,
        tgt_seq: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        return_prob: bool = False,
    ) -> torch.Tensor:
        del return_attn  # Kept for interface compatibility.

        x_left = self.sensorL_embed(sensorL).permute(0, 2, 1)
        x_right = self.sensorR_embed(sensorR).permute(0, 2, 1)
        x = self.pos_enc(x_left + x_right).permute(1, 0, 2)

        for layer in self.encoder:
            x = layer(x)
        memory = x

        y = self.pos_enc(self.tgt_embed(tgt_seq)).permute(1, 0, 2)
        if tgt_mask is None:
            tgt_mask = make_causal_mask(y.size(0), y.device)

        for layer in self.decoder:
            y = layer(y, memory, tgt_mask=tgt_mask)

        logits = self.classifier(self.dropout(y))
        if return_prob:
            return torch.softmax(logits, dim=-1)
        return logits
