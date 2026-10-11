#!/usr/bin/env python3
"""
Compare ESP32 exported 32-D normalized vectors against the same .tflite model
running on a PC with TensorFlow Lite / tflite-runtime.

Example:
  python validate_vectors.py --device http://192.168.4.1 \
      --model ../main/feature_extractor_32d_full_int8.tflite --max-patches 8

The ESP32 patch endpoint returns a 40x40 24-bit BMP. The script converts it
back to RGB888, applies the model's own input quantization parameters, invokes
the same int8 model on the host, dequantizes the output, L2-normalizes it, and
reports cosine similarity + max absolute error. It also compares nearest-
neighbor ranking for patch 0.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import tensorflow as tf  # type: ignore

    def make_interpreter(model_path: str):
        return tf.lite.Interpreter(model_path=model_path)
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Need TensorFlow or tflite-runtime. Install one of them, plus "
            "numpy, requests and pillow."
        ) from exc

    def make_interpreter(model_path: str):
        return Interpreter(model_path=model_path)


def get_json(session: requests.Session, url: str) -> dict:
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def load_esp32_patch(session: requests.Session, base: str, index: int) -> np.ndarray:
    r = session.get(f"{base}/api/ai_patch/patch", params={"index": index}, timeout=15)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != (40, 40, 3):
        raise RuntimeError(f"patch {index}: unexpected shape {arr.shape}")
    return arr


def quantize_patch(rgb: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    scale, zp = input_detail["quantization"]
    if dtype == np.int8:
        if scale <= 0:
            raise RuntimeError("Model int8 input has invalid scale")
        q = np.rint((rgb.astype(np.float32) / 255.0) / scale + zp)
        return np.clip(q, -128, 127).astype(np.int8)[None, ...]
    if dtype == np.uint8:
        if scale <= 0:
            return rgb.astype(np.uint8)[None, ...]
        q = np.rint((rgb.astype(np.float32) / 255.0) / scale + zp)
        return np.clip(q, 0, 255).astype(np.uint8)[None, ...]
    if dtype == np.float32:
        return (rgb.astype(np.float32) / 255.0)[None, ...].astype(np.float32)
    raise RuntimeError(f"Unsupported input dtype: {dtype}")


def host_vector(interpreter, input_detail: dict, output_detail: dict, rgb: np.ndarray) -> np.ndarray:
    inp = quantize_patch(rgb, input_detail)
    interpreter.set_tensor(input_detail["index"], inp)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_detail["index"])[0]

    if output_detail["dtype"] == np.int8:
        scale, zp = output_detail["quantization"]
        raw = (raw.astype(np.float32) - float(zp)) * float(scale)
    elif output_detail["dtype"] == np.uint8:
        scale, zp = output_detail["quantization"]
        raw = (raw.astype(np.float32) - float(zp)) * float(scale)
    else:
        raw = raw.astype(np.float32)

    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise RuntimeError("Host produced a zero vector")
    return raw / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="http://192.168.4.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-patches", type=int, default=8)
    ap.add_argument("--session-image", type=int, default=None,
                    help="0-based session image. Omit to use /api/ai_infer/features.")
    args = ap.parse_args()

    base = args.device.rstrip("/")
    session = requests.Session()

    patch_status = get_json(session, f"{base}/api/ai_patch/status")
    n = int(patch_status.get("count", patch_status.get("patch_count", 0)))
    if n <= 0:
        raise SystemExit(f"ESP32 reports no patches: {patch_status}")
    n = min(n, args.max_patches)

    if args.session_image is None:
        esp_features = get_json(session, f"{base}/api/ai_infer/features")
    else:
        esp_features = get_json(
            session,
            f"{base}/api/ai_session/features?image={args.session_image}",
        )

    vectors = esp_features.get("vectors")
    if not vectors:
        raise SystemExit("ESP32 response has no 'vectors' array")
    esp = np.asarray(vectors, dtype=np.float32)
    if esp.ndim != 2 or esp.shape[1] != 32:
        raise SystemExit(f"Unexpected ESP32 vector shape: {esp.shape}")
    n = min(n, esp.shape[0])

    model_path = str(Path(args.model).resolve())
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    print("Model input:", input_detail["shape"], input_detail["dtype"], input_detail["quantization"])
    print("Model output:", output_detail["shape"], output_detail["dtype"], output_detail["quantization"])
    print(f"Comparing {n} patch(es) ...")

    host = []
    sims = []
    max_abs = []
    for i in range(n):
        rgb = load_esp32_patch(session, base, i)
        hv = host_vector(interpreter, input_detail, output_detail, rgb)
        host.append(hv)
        sim = cosine(esp[i], hv)
        err = float(np.max(np.abs(esp[i] - hv)))
        sims.append(sim)
        max_abs.append(err)
        print(f"patch {i:3d}: cosine={sim:.7f}  max_abs={err:.7g}")

    host = np.asarray(host, dtype=np.float32)
    print("\nSUMMARY")
    print(f"min cosine     = {min(sims):.7f}")
    print(f"mean cosine    = {sum(sims)/len(sims):.7f}")
    print(f"max abs error  = {max(max_abs):.7g}")

    # Ranking check uses patch 0 as the query and is meaningful only when more
    # than one patch is available.
    if n > 1:
        esp_scores = esp[:n] @ esp[0]
        host_scores = host[:n] @ host[0]
        esp_rank = np.argsort(-esp_scores).tolist()
        host_rank = np.argsort(-host_scores).tolist()
        same_prefix = 0
        for a, b in zip(esp_rank, host_rank):
            if a != b:
                break
            same_prefix += 1
        print(f"ranking prefix match for query patch 0 = {same_prefix}/{n}")
        if esp_rank != host_rank:
            print("ESP rank:", esp_rank)
            print("PC  rank:", host_rank)

    # The user's practical gate: >= 0.99 is a strong pass indicator for this
    # int8-reference comparison. Do not call a near miss a perfect match.
    if min(sims) >= 0.99:
        print("RESULT: PASS (all cosine similarities >= 0.99)")
        return 0
    print("RESULT: CHECK (at least one cosine similarity < 0.99)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
