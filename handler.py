"""
RunPod Serverless Handler for FLUX.2-klein-base-9B with LoRA Support

This handler generates images using the FLUX.2-klein-base-9B model
with optional custom LoRA weights.

Based on ai-toolkit by ostris: https://github.com/ostris/ai-toolkit
"""

import base64
import io
import json
import os
import tempfile
import time
import urllib.request
import uuid
from typing import Dict, Any, List, Optional, Union

import numpy as np

# Must be set before CUDA initialises (before `import torch` triggers CUDA init).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["QUANTO_DISABLE_MARLIN"] = "1"

import boto3
import torch
from botocore.exceptions import ClientError
from diffusers import Flux2KleinPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageFilter


# RunPod serverless SDK
import runpod
from runpod.serverless.utils.rp_validator import validate


# ============================================================================
# S3 Configuration
# ============================================================================

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))

# HuggingFace Token for gated model access
HF_TOKEN = os.getenv("HF_TOKEN", "")


def get_s3_client():
    """Initialize and return S3 client."""
    if not S3_BUCKET_NAME or not S3_ACCESS_KEY_ID or not S3_SECRET_ACCESS_KEY:
        return None
    
    config = {
        "service_name": "s3",
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY_ID,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY,
    }
    
    if S3_ENDPOINT_URL:
        config["endpoint_url"] = S3_ENDPOINT_URL
    
    return boto3.client(**config)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.2-klein-base-9B")
DEFAULT_LORA_SCALE = float(os.getenv("DEFAULT_LORA_SCALE", "0.85"))
DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = os.getenv("DTYPE", "float8_e4m3fn").lower()
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "true").lower() == "true"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"

# Map dtype string to torch dtype
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float8": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
}

UPSCALER_MODEL_URL = "https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_drct-l/4xRealWebPhoto_v4_drct-l.pth"
UPSCALER_MODEL_PATH = os.getenv("UPSCALER_MODEL_PATH", "/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth")
UPSCALE_TILE_SIZE = 384
UPSCALE_TILE_OVERLAP = 64


# ============================================================================
# Optimized Presets
# ============================================================================

PRESETS = {
    "realistic_character": {
        "num_inference_steps": 35,
        "guidance_scale": 2.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1024,
        "description": "Photorealistic human portrait"
    },
    "portrait_hd": {
        "num_inference_steps": 40,
        "guidance_scale": 2.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1536,
        "description": "2:3 vertical portrait"
    },
    "manga_style": {
        "num_inference_steps": 25,
        "guidance_scale": 1.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1024,
        "description": "Matches ComfyUI Manga defaults - low CFG for style adherence"
    },
    "character_portrait_best": {
        "num_inference_steps": 45,
        "guidance_scale": 2.2,
        "shift": 2.5,
        "width": 1024,
        "height": 1024,
        "description": "Max fidelity character portraits"
    },
}


# ============================================================================
# Global Pipeline Instance
# ============================================================================

pipeline: Optional[Flux2KleinPipeline] = None
model_loaded = False
lora_adapters_loaded: List[Dict[str, str]] = []
upscaler_model = None


# ============================================================================
# Input Validation Schema
# ============================================================================

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": True},
    "preset": {"type": str, "required": False, "default": "realistic_character"},
    "width": {"type": int, "required": False, "default": 1024},
    "height": {"type": int, "required": False, "default": 1024},
    "num_inference_steps": {"type": int, "required": False, "default": 35},
    "guidance_scale": {"type": float, "required": False, "default": 2.0},
    "seed": {"type": int, "required": False, "default": -1},
    "num_images": {"type": int, "required": False, "default": 1},
    "loras": {"type": list, "required": False, "default": []},
    "lora_scale_mode": {"type": str, "required": False, "default": "absolute"},
    "output_format": {"type": str, "required": False, "default": "jpeg"},
    "return_type": {"type": str, "required": False, "default": "s3" if S3_BUCKET_NAME else "base64"},
    "max_sequence_length": {"type": int, "required": False, "default": 512},
    "shift": {"type": float, "required": False, "default": 1.5},
    "enable_2nd_pass": {"type": bool, "required": False, "default": False},
    "second_pass_strength": {"type": float, "required": False, "default": 0.2},
    "second_pass_steps": {"type": int, "required": False, "default": 12},
    "second_pass_guidance_scale": {"type": float, "required": False, "default": 1.0},
    "second_pass_lora_scale_multiplier": {"type": float, "required": False, "default": 1.0},
    "enable_upscale": {"type": bool, "required": False, "default": False},
    "upscale_factor": {"type": float, "required": False, "default": 2.0},
    "upscale_blend": {"type": float, "required": False, "default": 0.35},
    # Aliases
    "lora_path": {"type": str, "required": False, "default": ""},
    "lora_url": {"type": str, "required": False, "default": ""},
    "lora_scale": {"type": float, "required": False, "default": 0.85},
    "additional_lora": {"type": str, "required": False, "default": ""},
    "additional_lora_path": {"type": str, "required": False, "default": ""},
    "additional_lora_url": {"type": str, "required": False, "default": ""},
}


# ============================================================================
# Core Helper Logic
# ============================================================================

def encode_image_to_base64(image: Image.Image, format: str = "jpeg") -> str:
    buffer = io.BytesIO()
    if format.lower() == "jpeg": image.save(buffer, format="JPEG", quality=95, optimize=True)
    else: image.save(buffer, format=format.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def upload_to_s3(image: Image.Image, format: str = "jpeg") -> Optional[str]:
    s3 = get_s3_client()
    if not s3: return None
    ext = "jpg" if format.lower() == "jpeg" else format.lower()
    key = f"flux2-klein/{uuid.uuid4()}.{ext}"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG" if ext == "jpg" else format.upper(), quality=95)
    buffer.seek(0)
    try:
        s3.upload_fileobj(buffer, S3_BUCKET_NAME, key, ExtraArgs={"ContentType": f"image/{ext}"})
        return s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET_NAME, "Key": key}, ExpiresIn=S3_PRESIGNED_URL_EXPIRY)
    except: return None


def load_upscaler():
    global upscaler_model
    if upscaler_model is not None: return upscaler_model
    from spandrel import ModelLoader
    if not os.path.exists(UPSCALER_MODEL_PATH):
        os.makedirs(os.path.dirname(UPSCALER_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(UPSCALER_MODEL_URL, UPSCALER_MODEL_PATH)
    upscaler_model = ModelLoader().load_from_file(UPSCALER_MODEL_PATH).cuda().eval()
    return upscaler_model


def tiled_upscale(image: Image.Image, factor: float, blend: float) -> Image.Image:
    model = load_upscaler()
    scale = model.scale
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    in_h, in_w = tensor.shape[2], tensor.shape[3]
    out_h, out_w = in_h * scale, in_w * scale
    output, weight = torch.zeros((1, 3, out_h, out_w)), torch.zeros((1, 1, out_h, out_w))
    step = UPSCALE_TILE_SIZE - UPSCALE_TILE_OVERLAP
    
    y_starts = list(range(0, max(1, in_h - UPSCALE_TILE_SIZE), step)) + [max(0, in_h - UPSCALE_TILE_SIZE)]
    x_starts = list(range(0, max(1, in_w - UPSCALE_TILE_SIZE), step)) + [max(0, in_w - UPSCALE_TILE_SIZE)]
    
    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            tile = tensor[:, :, y:y+UPSCALE_TILE_SIZE, x:x+UPSCALE_TILE_SIZE].to(DEVICE)
            with torch.inference_mode(): tile_out = model(tile).cpu().clamp(0.0, 1.0)
            oy, ox = y * scale, x * scale
            th, tw = tile_out.shape[2], tile_out.shape[3]
            output[:, :, oy:oy+th, ox:ox+tw] += tile_out
            weight[:, :, oy:oy+th, ox:ox+tw] += 1.0
            
    res = (output / weight.clamp(min=1e-8)).clamp(0.0, 1.0)
    res_img = Image.fromarray((res.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8))
    target_w, target_h = int(round(in_w * factor)), int(round(in_h * factor))
    if (target_w, target_h) != (out_w, out_h): res_img = res_img.resize((target_w, target_h), Image.LANCZOS)
    if blend < 1.0:
        base = image.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        res_img = Image.blend(base, res_img, alpha=blend)
    return res_img


def _set_active_lora_adapters(pipeline, adapters, mode="absolute", multiplier=1.0):
    if not adapters: return []
    scales = [float(item["scale"]) * multiplier for item in adapters]
    if mode == "normalized":
        total = sum(scales)
        if total > 0: scales = [s / total for s in scales]
    
    applied = []
    for item, eff in zip(adapters, scales):
        applied.append({"adapter_name": item["adapter_name"], "effective_scale": eff, "path": item["path"], "scale": item["scale"]})
    
    names = [x["adapter_name"] for x in applied]
    weights = [x["effective_scale"] for x in applied]
    
    for comp_name in ["transformer", "text_encoder", "text_encoder_2"]:
        comp = getattr(pipeline, comp_name, None)
        if comp is not None and hasattr(comp, "set_adapters"):
            try:
                comp.set_adapters(names, adapter_weights=weights)
                # Aggressive manual injection for stylistic adherence
                for m in comp.modules():
                    if hasattr(m, "scaling") and hasattr(m, "adapter_name") and m.adapter_name in names:
                        m.scaling[m.adapter_name] = weights[names.index(m.adapter_name)]
            except: pass
    return applied


def initialize_pipeline(model_id, adapters=None):
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Match ComfyUI fidelity: 24GB (4090) uses FP8, 40GB+ uses BF16, <24GB uses INT8
    if vram >= 40.0: target_dtype, use_quant, q_type = torch.bfloat16, False, None
    elif vram >= 22.0: target_dtype, use_quant, q_type = torch.float8_e4m3fn, True, torch.float8_e4m3fn
    else: target_dtype, use_quant, q_type = torch.float8_e4m3fn, True, torch.int8

    print(f"Loading {model_id} in {target_dtype} (VRAM: {vram:.1f}GB, Quant: {use_quant})")
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16 if use_quant else target_dtype, token=HF_TOKEN or None)

    if use_quant:
        from optimum.quanto import freeze, qint8, quantize
        q_weights = qint8 if q_type == torch.int8 else q_type
        print(f"Quantizing to {q_weights}...")
        quantize(pipeline.transformer, weights=q_weights)
        freeze(pipeline.transformer)

    loaded = []
    for l in (adapters or []):
        print(f"Loading LoRA: {l['path']} as '{l['adapter_name']}'")
        try:
            if l['path'].startswith("http"):
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "lora.safetensors")
                    urllib.request.urlretrieve(l['path'], path)
                    pipeline.load_lora_weights(tmp, weight_name="lora.safetensors", adapter_name=l['adapter_name'])
            else: pipeline.load_lora_weights(l['path'], adapter_name=l['adapter_name'])
            loaded.append(l)
        except Exception as e: print(f"Error loading LoRA: {e}")
    
    _set_active_lora_adapters(pipeline, loaded)
    if ENABLE_CPU_OFFLOAD: pipeline.enable_model_cpu_offload()
    else: pipeline.to(DEVICE)
    pipeline.vae.enable_slicing(); pipeline.vae.enable_tiling()
    return pipeline, loaded


def generate_images(ji, ef):
    global pipeline, lora_adapters_loaded
    # Normalize LoRAs
    req_loras = []
    if "loras" in ef:
        for i, l in enumerate(ji.get("loras", [])):
            p = l.get("path") or l.get("url") or l.get("lora_url") or l.get("lora_path")
            s = l.get("scale") or l.get("strength") or l.get("weight") or 0.85
            n = l.get("adapter_name") or l.get("name") or f"lora_{i}"
            if p: req_loras.append({"path": p, "scale": float(s), "adapter_name": n})
    else:
        lp = (ji.get("lora_path") or ji.get("lora_url") or "").strip()
        if lp: req_loras.append({"path": lp, "scale": float(ji.get("lora_scale", 0.85)), "adapter_name": "lora_0"})

    # Clean reinit on LoRA change
    sig = lambda x: tuple((y["path"], y["adapter_name"]) for y in x)
    if pipeline is None or sig(req_loras) != sig(lora_adapters_loaded):
        import gc; pipeline = None; gc.collect(); torch.cuda.empty_cache()
        pipeline, lora_adapters_loaded = initialize_pipeline(ji.get("model_id", DEFAULT_MODEL_ID), req_loras)

    p = PRESETS.get(ji.get("preset"), PRESETS["realistic_character"]).copy()
    w, h = ji.get("width", p["width"]), ji.get("height", p["height"])
    steps = ji.get("num_inference_steps", p["num_inference_steps"])
    cfg = ji.get("guidance_scale", p["guidance_scale"])
    shift = ji.get("shift", p.get("shift", 1.5))
    seed = ji.get("seed", -1)
    if seed < 0: seed = int(time.time() * 1000) % (2**31)
    
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    applied = _set_active_lora_adapters(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
    
    print(f"Inference: {w}x{h}, {steps} steps, CFG {cfg}")
    start = time.time()
    with torch.inference_mode():
        res = pipeline(prompt=ji["prompt"], width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator(DEVICE).manual_seed(seed), num_images_per_prompt=ji.get("num_images", 1), max_sequence_length=ji.get("max_sequence_length", 512))
    
    final = []
    for img in res.images:
        if ji.get("enable_2nd_pass"):
            _set_active_lora_adapters(pipeline, lora_adapters_loaded, multiplier=ji.get("second_pass_lora_scale_multiplier", 1.0))
            with torch.inference_mode():
                refined = pipeline(prompt=ji["prompt"], image=img, num_inference_steps=ji.get("second_pass_steps", 12), guidance_scale=ji.get("second_pass_guidance_scale", 1.0)).images[0]
            img = _transfer_high_frequency_details(img, refined, ji.get("second_pass_strength", 0.2))
        if ji.get("enable_upscale"):
            img = tiled_upscale(img, ji.get("upscale_factor", 2.0), ji.get("upscale_blend", 0.35))
        final.append(img)

    out_fmt = ji.get("output_format", "jpeg")
    rt = ji.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    meta = {"loras": applied, "generation_time": f"{time.time()-start:.2f}s", "seed": seed}
    
    if rt == "s3":
        urls = [upload_to_s3(i, out_fmt) or encode_image_to_base64(i, out_fmt) for i in final]
        return {"image_urls": urls, "metadata": meta}
    return {"images": [encode_image_to_base64(i, out_fmt) for i in final], "metadata": meta}


def _transfer_high_frequency_details(base, refined, amount):
    b_np = np.asarray(base.convert("RGB"), dtype=np.float32)
    r_rgb = refined.convert("RGB")
    r_low = np.asarray(r_rgb.filter(ImageFilter.GaussianBlur(1.25)), dtype=np.float32)
    diff = (np.asarray(r_rgb, dtype=np.float32) - r_low) * amount
    return Image.fromarray(np.clip(b_np + diff, 0, 255).astype(np.uint8))


def handler(job):
    try:
        raw = job.get("input", {})
        ji = _preprocess_job_input(raw)
        val = validate(ji, INPUT_SCHEMA)
        if "errors" in val: return {"error": val["errors"]}
        return generate_images(val["validated_input"], set(raw.keys()))
    except Exception as e:
        import traceback; return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
