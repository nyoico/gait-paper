#!/usr/bin/env python3
"""ESP32-S3 sample receiver with a reliable READY handshake and optional ONNX inference.

Run on Raspberry Pi:
    python pi_receive_fixed.py

No command-line parser is used. Edit only the USER SETTINGS section.
"""

from __future__ import annotations

import json
import re
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import serial


# =============================================================================
# USER SETTINGS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

SERIAL_PORT = "/dev/ttyACM0"  # Change to /dev/ttyUSB0 if that is the real port.
BAUD_RATE = 115200
SERIAL_STARTUP_DELAY_SECONDS = 2.5
PACKET_TIMEOUT_SECONDS = 5.0
READY_RETRY_SECONDS = 0.5
WAIT_STATUS_SECONDS = 5.0

EXPECTED_TIME_STEPS = 101
SAVE_RECEIVED_PACKET = True
SAVE_DIR = BASE_DIR / "received"

# First test the ESP32 -> Raspberry Pi link with False.
USE_MODEL_INFERENCE = False

# This code loads ONNX with ONNX Runtime. It does NOT call torch.load().
MODEL_PATH = BASE_DIR / "student_qkd_int8.onnx"
NORMALIZATION_PATH = BASE_DIR / "normalization.npz"
ONNX_THREADS = 2

# Normally leave these as None. Set exact names only when automatic detection fails.
# The script prints every ONNX input name when it loads the model.
ONNX_LEFT_INPUT_NAME: str | None = None
ONNX_RIGHT_INPUT_NAME: str | None = None
ONNX_COMBINED_INPUT_NAME: str | None = None
ONNX_TGT_INPUT_NAME: str | None = None
ONNX_MASK_INPUT_NAME: str | None = None
# =============================================================================


MAGIC = b"GRF1"
VERSION = 1
HEADER = struct.Struct("<4sHHIHHBBHIhHI")  # 32 bytes
DTYPE_FLOAT32 = 1
LAYOUT_TIME_CHANNEL = 1
FLAG_NORMALIZED = 0x0001

HOST_READY = bytes([0x52])  # 'R'
ESP_READY = 0x53            # Optional status byte supported by future sender revisions.
ACK_BYTE = bytes([0x06])
NAK_BYTE = bytes([0x15])
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024

VALID_CLASSES = [
    "HC", "H_P", "H_C", "H_F",
    "K_P", "K_F", "K_R",
    "A_F", "A_R", "A_L",
    "C_F", "C_A",
]

FINE_TO_COARSE = {
    "HC": "HC",
    "H_P": "H",
    "H_C": "H",
    "H_F": "H",
    "K_P": "K",
    "K_F": "K",
    "K_R": "K",
    "A_F": "A",
    "A_R": "A",
    "A_L": "A",
    "C_F": "C",
    "C_A": "C",
}


@dataclass(frozen=True)
class Packet:
    sample_id: int
    label_index: int
    normalized: bool
    data_tc: np.ndarray  # [T, 10]

    @property
    def left_ct(self) -> np.ndarray:
        return np.ascontiguousarray(self.data_tc[:, :5].T, dtype=np.float32)

    @property
    def right_ct(self) -> np.ndarray:
        return np.ascontiguousarray(self.data_tc[:, 5:10].T, dtype=np.float32)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {description}:\n{path.resolve()}\n"
            "Check the USER SETTINGS section."
        )


def send_control_byte(ser: serial.Serial, value: bytes) -> None:
    ser.write(value)
    ser.flush()


def read_exact(ser: serial.Serial, size: int, timeout_s: float) -> bytes:
    if size < 0:
        raise ValueError("size must be non-negative")

    data = bytearray()
    deadline = time.monotonic() + timeout_s

    while len(data) < size:
        chunk = ser.read(size - len(data))
        if chunk:
            data.extend(chunk)
            continue

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Receive timeout: received only {len(data)}/{size} bytes"
            )

    return bytes(data)


def find_magic_with_ready(ser: serial.Serial) -> bytes:
    """Wait for GRF1 while repeatedly advertising HOST_READY.

    The previous implementation sent HOST_READY only once. If opening the serial
    port reset the ESP32, or if ESP32 was still mounting the SD card, that byte
    could be missed and the sender would keep hostReady=False forever.
    """
    window = bytearray()
    next_ready = 0.0
    next_status = time.monotonic() + WAIT_STATUS_SECONDS
    esp_ready_reported = False

    while True:
        now = time.monotonic()

        if now >= next_ready:
            send_control_byte(ser, HOST_READY)
            next_ready = now + READY_RETRY_SECONDS

        chunk = ser.read(1)
        if chunk:
            value = chunk[0]
            if value == ESP_READY:
                if not esp_ready_reported:
                    print("[LINK] ESP32 acknowledged HOST_READY.")
                    esp_ready_reported = True
                continue

            window.extend(chunk)
            if len(window) > len(MAGIC):
                del window[0]
            if bytes(window) == MAGIC:
                return MAGIC

        now = time.monotonic()
        if now >= next_status:
            print(
                "[WAIT] No packet yet. HOST_READY is being retransmitted. "
                "Press BOOT briefly after the ESP32 has booted."
            )
            next_status = now + WAIT_STATUS_SECONDS


def receive_packet(ser: serial.Serial, timeout_s: float) -> Packet:
    prefix = find_magic_with_ready(ser)
    header_bytes = prefix + read_exact(ser, HEADER.size - len(prefix), timeout_s)

    (
        magic,
        version,
        header_size,
        sample_id,
        time_steps,
        channels,
        dtype_code,
        layout_code,
        flags,
        payload_size,
        label_index,
        _reserved,
        expected_crc,
    ) = HEADER.unpack(header_bytes)

    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"Unsupported packet version: {version}")
    if header_size != HEADER.size:
        raise ValueError(f"Invalid header size: {header_size}")
    if dtype_code != DTYPE_FLOAT32:
        raise ValueError(f"Unsupported dtype code: {dtype_code}")
    if layout_code != LAYOUT_TIME_CHANNEL:
        raise ValueError(f"Unsupported layout code: {layout_code}")
    if channels != 10:
        raise ValueError(f"Expected 10 channels, received {channels}")
    if time_steps != EXPECTED_TIME_STEPS:
        raise ValueError(
            f"Expected {EXPECTED_TIME_STEPS} time steps, received {time_steps}"
        )

    calculated_size = int(time_steps) * int(channels) * 4
    if payload_size != calculated_size:
        raise ValueError(
            f"Payload size={payload_size}, expected={calculated_size} from shape"
        )
    if payload_size <= 0 or payload_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"Invalid payload size: {payload_size}")

    payload = read_exact(ser, payload_size, timeout_s)
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            f"CRC mismatch: header=0x{expected_crc:08x}, "
            f"calculated=0x{actual_crc:08x}"
        )

    data_tc = np.frombuffer(payload, dtype="<f4").reshape(time_steps, channels).copy()
    if not np.isfinite(data_tc).all():
        raise ValueError("Payload contains NaN or Inf")

    return Packet(
        sample_id=int(sample_id),
        label_index=int(label_index),
        normalized=bool(flags & FLAG_NORMALIZED),
        data_tc=data_tc,
    )


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _numpy_dtype_for_onnx(type_name: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(float16)": np.dtype(np.float16),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(bool)": np.dtype(np.bool_),
        "tensor(uint8)": np.dtype(np.uint8),
        "tensor(int8)": np.dtype(np.int8),
    }
    if type_name not in mapping:
        raise TypeError(f"Unsupported ONNX input type: {type_name}")
    return mapping[type_name]


class OnnxInferenceEngine:
    def __init__(self, model_path: Path, normalization_path: Path, threads: int) -> None:
        require_file(model_path, "ONNX model")
        require_file(normalization_path, "normalization statistics")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX inference requires onnxruntime. Install it with: "
                "python -m pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.inputs = list(self.session.get_inputs())
        self.outputs = list(self.session.get_outputs())

        print("ONNX inputs:")
        for item in self.inputs:
            print(f"  - name={item.name!r}, shape={item.shape}, type={item.type}")
        print("ONNX outputs:")
        for item in self.outputs:
            print(f"  - name={item.name!r}, shape={item.shape}, type={item.type}")
        if not self.outputs:
            raise ValueError("The ONNX model has no outputs")
        if self.outputs[0].type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
            raise TypeError(
                "The first ONNX output is an integer tensor. Exact output "
                "dequantization scale/zero-point are required before softmax."
            )

        norm = np.load(normalization_path)
        try:
            self.left_mean = self._validate_norm(norm["left_mean"], "left_mean")
            self.left_std = self._validate_norm(norm["left_std"], "left_std")
            self.right_mean = self._validate_norm(norm["right_mean"], "right_mean")
            self.right_std = self._validate_norm(norm["right_std"], "right_std")
            self.eps = float(np.asarray(norm["eps"]).item())
        finally:
            norm.close()

        self.input_by_name = {item.name: item for item in self.inputs}
        self.left_name = self._resolve_input_name(
            ONNX_LEFT_INPUT_NAME, ("sensorl", "left", "inputl", "xleft")
        )
        self.right_name = self._resolve_input_name(
            ONNX_RIGHT_INPUT_NAME, ("sensorr", "right", "inputr", "xright")
        )
        self.combined_name = self._resolve_input_name(
            ONNX_COMBINED_INPUT_NAME, ("combined", "sensor", "input", "x"),
            required=False,
            exclude={self.left_name, self.right_name},
        )
        self.mask_name = self._resolve_input_name(
            ONNX_MASK_INPUT_NAME, ("tgtmask", "mask"), required=False
        )
        self.tgt_name = self._resolve_input_name(
            ONNX_TGT_INPUT_NAME, ("tgtseq", "target", "tgt", "token", "decoder"),
            required=False,
            exclude={self.mask_name},
        )

        # Fallback for the common export order: left, right, tgt_seq, tgt_mask.
        float_rank3 = [
            item for item in self.inputs
            if item.type in {"tensor(float)", "tensor(float16)", "tensor(double)"}
            and len(item.shape) == 3
        ]
        if self.left_name is None and self.right_name is None and len(float_rank3) >= 2:
            self.left_name = float_rank3[0].name
            self.right_name = float_rank3[1].name
            print(
                "[WARN] Left/right input names were inferred from ONNX input order: "
                f"{self.left_name!r}, {self.right_name!r}"
            )

        if self.left_name is None or self.right_name is None:
            # Support a single combined [1,10,T] or [1,T,10] input.
            if self.combined_name is None:
                float_rank3_names = [item.name for item in float_rank3]
                if len(float_rank3_names) == 1:
                    self.combined_name = float_rank3_names[0]
                else:
                    raise ValueError(
                        "Could not identify ONNX left/right inputs. Set "
                        "ONNX_LEFT_INPUT_NAME and ONNX_RIGHT_INPUT_NAME at the top."
                    )

        recognized = {
            name for name in (
                self.left_name,
                self.right_name,
                self.combined_name,
                self.tgt_name,
                self.mask_name,
            )
            if name is not None
        }
        unknown = [item.name for item in self.inputs if item.name not in recognized]
        if unknown:
            raise ValueError(
                "Unmapped ONNX inputs: " + ", ".join(repr(name) for name in unknown)
                + ". Set their names in USER SETTINGS or adjust the exporter."
            )

    @staticmethod
    def _validate_norm(value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (1, 5, 1):
            raise ValueError(f"{name} must have shape (1,5,1), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
        return value

    def _resolve_input_name(
        self,
        explicit: str | None,
        tokens: tuple[str, ...],
        *,
        required: bool = False,
        exclude: set[str | None] | None = None,
    ) -> str | None:
        excluded = {name for name in (exclude or set()) if name is not None}

        if explicit is not None:
            if explicit not in self.input_by_name:
                raise KeyError(
                    f"Configured ONNX input {explicit!r} does not exist. "
                    f"Available={list(self.input_by_name)}"
                )
            return explicit

        matches = []
        for item in self.inputs:
            if item.name in excluded:
                continue
            normalized = _normalize_name(item.name)
            if any(token in normalized for token in tokens):
                matches.append(item.name)

        if len(matches) == 1:
            return matches[0]
        if required and not matches:
            raise ValueError(f"Could not resolve ONNX input for tokens={tokens}")
        return None

    @staticmethod
    def _cast_for_input(value: np.ndarray, input_meta: Any) -> np.ndarray:
        dtype = _numpy_dtype_for_onnx(input_meta.type)
        return np.ascontiguousarray(value, dtype=dtype)

    def _build_feeds(self, left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}

        if self.left_name is not None and self.right_name is not None:
            left_meta = self.input_by_name[self.left_name]
            right_meta = self.input_by_name[self.right_name]
            allowed_sensor_types = {
                "tensor(float)", "tensor(float16)", "tensor(double)"
            }
            if left_meta.type not in allowed_sensor_types or right_meta.type not in allowed_sensor_types:
                raise TypeError(
                    "The ONNX sensor inputs are integer tensors. This receiver cannot "
                    "guess their quantization scale/zero-point. Export a model with "
                    "float external inputs and internal Q/DQ nodes, or add the exact "
                    "input quantization parameters."
                )
            feeds[self.left_name] = self._cast_for_input(left, left_meta)
            feeds[self.right_name] = self._cast_for_input(right, right_meta)
        elif self.combined_name is not None:
            meta = self.input_by_name[self.combined_name]
            if meta.type not in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
                raise TypeError(
                    "The combined ONNX sensor input is an integer tensor. Exact "
                    "quantization scale/zero-point are required; raw casting is invalid."
                )
            combined_ct = np.concatenate([left, right], axis=1)  # [1,10,T]
            shape = meta.shape
            if len(shape) != 3:
                raise ValueError(
                    f"Combined ONNX input must be rank 3, got {shape}"
                )
            if shape[-1] == 10:
                combined = np.transpose(combined_ct, (0, 2, 1))  # [1,T,10]
            else:
                combined = combined_ct
            feeds[self.combined_name] = self._cast_for_input(combined, meta)
        else:
            raise RuntimeError("No usable ONNX sensor input mapping")

        if self.tgt_name is not None:
            meta = self.input_by_name[self.tgt_name]
            tgt = np.zeros((1, 1), dtype=_numpy_dtype_for_onnx(meta.type))
            feeds[self.tgt_name] = np.ascontiguousarray(tgt)

        if self.mask_name is not None:
            meta = self.input_by_name[self.mask_name]
            mask = np.zeros((1, 1), dtype=_numpy_dtype_for_onnx(meta.type))
            feeds[self.mask_name] = np.ascontiguousarray(mask)

        return feeds

    def predict(self, packet: Packet) -> dict[str, Any]:
        if packet.data_tc.shape != (EXPECTED_TIME_STEPS, 10):
            raise ValueError(
                f"Expected [{EXPECTED_TIME_STEPS},10], got {packet.data_tc.shape}"
            )

        left = packet.left_ct[None, ...]   # [1,5,T]
        right = packet.right_ct[None, ...]  # [1,5,T]

        if not packet.normalized:
            left = (left - self.left_mean) / (self.left_std + self.eps)
            right = (right - self.right_mean) / (self.right_std + self.eps)

        feeds = self._build_feeds(left, right)

        start = time.perf_counter()
        output_values = self.session.run(None, feeds)
        latency_ms = (time.perf_counter() - start) * 1000.0

        if not output_values:
            raise RuntimeError("ONNX model returned no outputs")

        raw = np.asarray(output_values[0])
        if raw.size == 0 or raw.ndim == 0:
            raise ValueError(f"Unexpected ONNX output shape: {raw.shape}")

        class_dim = raw.shape[-1]
        if class_dim != len(VALID_CLASSES):
            raise ValueError(
                f"Expected {len(VALID_CLASSES)} class logits, got output shape {raw.shape}"
            )

        logits = raw.reshape(-1, class_dim)[-1].astype(np.float64, copy=False)
        if not np.isfinite(logits).all():
            raise ValueError("ONNX output contains NaN/Inf")

        # Accept either raw logits or an ONNX wrapper that already outputs
        # probabilities. Do not apply softmax twice.
        if np.all(logits >= 0.0) and np.isclose(logits.sum(), 1.0, rtol=1e-4, atol=1e-5):
            probabilities = logits / logits.sum()
        else:
            shifted = logits - logits.max()
            exp_values = np.exp(shifted)
            probabilities = exp_values / exp_values.sum()

        predicted_index = int(np.argmax(probabilities))
        fine_label = VALID_CLASSES[predicted_index]
        top_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "index": int(index),
                "label": VALID_CLASSES[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in top_indices
        ]

        return {
            "predicted_index": predicted_index,
            "fine_label": fine_label,
            "coarse_label": FINE_TO_COARSE[fine_label],
            "confidence": float(probabilities[predicted_index]),
            "latency_ms": latency_ms,
            "top3": top3,
        }


def save_packet(
    packet: Packet,
    save_dir: Path,
    result: dict[str, Any] | None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"sample_{packet.sample_id:06d}.npz"

    metadata = {
        "sample_id": packet.sample_id,
        "label_index": packet.label_index,
        "normalized": packet.normalized,
        "inference": result,
    }

    np.savez(
        path,
        data_tc=packet.data_tc,
        left_ct=packet.left_ct,
        right_ct=packet.right_ct,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def main() -> None:
    engine: OnnxInferenceEngine | None = None

    if USE_MODEL_INFERENCE:
        engine = OnnxInferenceEngine(MODEL_PATH, NORMALIZATION_PATH, ONNX_THREADS)
        print("ONNX model and normalization statistics loaded.")
    else:
        print("Receive-only mode: USE_MODEL_INFERENCE=False")

    print(f"Opening serial port: {SERIAL_PORT}")

    try:
        with serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=0.1,
            write_timeout=1.0,
        ) as ser:
            # Opening a serial port may reset the ESP32. Wait for SD initialization,
            # then continue retransmitting HOST_READY while no packet is received.
            time.sleep(SERIAL_STARTUP_DELAY_SECONDS)
            ser.reset_input_buffer()
            send_control_byte(ser, HOST_READY)

            print(
                "Raspberry Pi is ready. Wait for ESP32 boot, then press BOOT briefly."
            )

            while True:
                try:
                    packet = receive_packet(ser, PACKET_TIMEOUT_SECONDS)

                    # Save the verified raw packet first. This is quick and ensures
                    # the sample is not lost even if model inference later fails.
                    saved_path: Path | None = None
                    if SAVE_RECEIVED_PACKET:
                        saved_path = save_packet(packet, SAVE_DIR, result=None)

                    # ACK means transport acceptance (header, shape, CRC and optional
                    # raw save succeeded). Do not delay ACK for model inference.
                    send_control_byte(ser, ACK_BYTE)

                    print(
                        f"[OK] id={packet.sample_id}, shape={packet.data_tc.shape}, "
                        f"label={packet.label_index}, normalized={packet.normalized}"
                    )
                    if saved_path is not None:
                        print(f"     saved={saved_path.resolve()}")

                    if engine is not None:
                        try:
                            result = engine.predict(packet)
                            print(
                                f"     prediction={result['fine_label']} "
                                f"({result['coarse_label']}), "
                                f"confidence={result['confidence']:.4f}, "
                                f"latency={result['latency_ms']:.2f} ms"
                            )
                            print("     top3:")
                            for rank, item in enumerate(result["top3"], start=1):
                                print(
                                    f"       {rank}. {item['label']}: "
                                    f"{item['probability']:.4f}"
                                )

                            if SAVE_RECEIVED_PACKET:
                                save_packet(packet, SAVE_DIR, result=result)
                        except Exception as exc:
                            print(
                                f"[INFERENCE ERROR] {type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )

                except KeyboardInterrupt:
                    print("\nExiting.")
                    return
                except (serial.SerialException, serial.SerialTimeoutException):
                    raise
                except Exception as exc:
                    print(f"[PACKET ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
                    try:
                        send_control_byte(ser, NAK_BYTE)
                    except (serial.SerialException, serial.SerialTimeoutException):
                        raise

    except (serial.SerialException, serial.SerialTimeoutException) as exc:
        raise RuntimeError(
            f"Serial communication failed on {SERIAL_PORT}: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
