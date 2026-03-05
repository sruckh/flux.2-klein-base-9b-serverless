"""
RunPod Serverless Handler for FLUX.2-klein-base-9B with LoRA Support
Based on ai-toolkit by ostris
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
import boto3
import torch
from botocore.exceptions import ClientError
from diffusers import Flux2KleinPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageFilter

import runpod
from runpod.serverless.utils.rp_validator import validate

# Environment Config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["QUANTO_DISABLE_MARLIN"] = "1"

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.2-klein-base-9B")
DEVICE = os.getenv("DEVICE", "cuda")
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"

# Globals
pipeline = None
lora_adapters_loaded = []

PRESETS = {
    "realistic_character": {"num_inference_steps": 35, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1024},
    "manga_style": {"num_inference_steps": 25, "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1024},
    "portrait_hd": {"num_inference_steps": 40, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1536},
}

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
    "return_type": {"type": str, "required": False, "default": "s3"},
    "shift": {"type": float, "required": False, "default": 1.5},
    "enable_2nd_pass": {"type": bool, "required": False, "default": False},
    "second_pass_strength": {"type": float, "required": False, "default": 0.2},
    "second_pass_steps": {"type": int, "required": False, "default": 12},
    "second_pass_guidance_scale": {"type": float, "required": False, "default": 1.0},
    "second_pass_lora_scale_multiplier": {"type": float, "required": False, "default": 1.0},
}

def _preprocess_job_input(ji: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ji, dict): return ji
    ji = dict(ji)
    for k in ["lora_scale", "additional_lora_strength", "additional_lora_scale", "second_pass_lora_scale_multiplier"]:
        if k in ji and isinstance(ji[k], str) and ji[k].strip():
            try: ji[k] = float(ji[k].strip())
            except: pass
    if isinstance(ji.get("loras"), list):
        for item in ji["loras"]:
            if isinstance(item, dict):
                for sk in ("scale", "strength", "weight", "lora_scale"):
                    if sk in item and isinstance(item[sk], str) and item[sk].strip():
                        try: item[sk] = float(item[sk].strip())
                        except: pass
    return ji

def get_s3_client():
    if not S3_BUCKET_NAME: return None
    return boto3.client("s3", region_name=S3_REGION, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY, endpoint_url=S3_ENDPOINT_URL or None)

def upload_to_s3(image, fmt):
    s3 = get_s3_client()
    if not s3: return None
    buf = io.BytesIO()
    image.save(buf, format="JPEG" if fmt.lower()=="jpeg" else fmt.upper(), quality=95)
    buf.seek(0)
    key = f"flux2-klein/{uuid.uuid4()}.{fmt}"
    try:
        s3.upload_fileobj(buf, S3_BUCKET_NAME, key, ExtraArgs={"ContentType": f"image/{fmt}"})
        return s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET_NAME, "Key": key}, ExpiresIn=S3_PRESIGNED_URL_EXPIRY)
    except: return None

def encode_base64(image, fmt):
    buf = io.BytesIO()
    image.save(buf, format="JPEG" if fmt.lower()=="jpeg" else fmt.upper(), quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _set_loras(pipe, adapters, mode="absolute", multiplier=1.0):
    if not adapters: return []
    scales = [float(a["scale"]) * multiplier for a in adapters]
    if mode == "normalized" and sum(scales) > 0:
        total = sum(scales)
        scales = [s / total for s in scales]
    
    names = [a["adapter_name"] for a in adapters]
    applied = []
    for i, (name, scale) in enumerate(zip(names, scales)):
        applied.append({"adapter_name": name, "effective_scale": scale, "path": adapters[i]["path"]})
        
    for comp_name in ["transformer", "text_encoder", "text_encoder_2"]:
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "set_adapters"):
            try:
                comp.set_adapters(names, adapter_weights=scales)
                for m in comp.modules():
                    if hasattr(m, "scaling") and hasattr(m, "adapter_name") and m.adapter_name in names:
                        m.scaling[m.adapter_name] = scales[names.index(m.adapter_name)]
            except: pass
    return applied

def initialize_pipeline(model_id, adapters=None):
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # STYLE CHOICE: Use FP8 (qfloat8) for 24GB GPUs to match ComfyUI. Use BF16 for 40GB+.
    use_quant = vram < 40.0
    print(f"Loading {model_id} (VRAM: {vram:.1f}GB, Quantized: {use_quant})")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=HF_TOKEN or None)

    if use_quant:
        from optimum.quanto import freeze, qint8, qfloat8, quantize
        # On 24GB (4090), qfloat8 is the best choice for style preservation.
        q_weights = qfloat8 if vram >= 22.0 else qint8
        print(f"Quantizing transformer to {q_weights}...")
        quantize(pipe.transformer, weights=q_weights)
        freeze(pipe.transformer)

    loaded = []
    for l in (adapters or []):
        print(f"Loading LoRA: {l['path']} as '{l['adapter_name']}'")
        try:
            if l['path'].startswith("http"):
                with tempfile.TemporaryDirectory() as tmp:
                    p = os.path.join(tmp, "lora.safetensors")
                    urllib.request.urlretrieve(l['path'], p)
                    pipe.load_lora_weights(tmp, weight_name="lora.safetensors", adapter_name=l['adapter_name'])
            else: pipe.load_lora_weights(l['path'], adapter_name=l['adapter_name'])
            loaded.append(l)
        except Exception as e: print(f"LoRA Error: {e}")
    
    _set_loras(pipe, loaded)
    if ENABLE_CPU_OFFLOAD: pipe.enable_model_cpu_offload()
    else: pipe.to(DEVICE)
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
    return pipe, loaded

def generate_images(ji, ef):
    global pipeline, lora_adapters_loaded
    req_loras = []
    if "loras" in ef:
        for i, l in enumerate(ji.get("loras", [])):
            p = l.get("path") or l.get("url") or l.get("lora_url") or l.get("lora_path")
            if p: req_loras.append({"path": p, "scale": float(l.get("scale", 0.85)), "adapter_name": l.get("adapter_name", f"l_{i}")})
    
    sig = lambda x: tuple((y["path"], y["adapter_name"]) for y in x)
    if pipeline is None or sig(req_loras) != sig(lora_adapters_loaded):
        import gc; pipeline = None; gc.collect(); torch.cuda.empty_cache()
        pipeline, lora_adapters_loaded = initialize_pipeline(ji.get("model_id", DEFAULT_MODEL_ID), req_loras)

    preset = PRESETS.get(ji.get("preset"), PRESETS["realistic_character"])
    w = ji.get("width") if "width" in ef else preset["width"]
    h = ji.get("height") if "height" in ef else preset["height"]
    steps = ji.get("num_inference_steps") if "num_inference_steps" in ef else preset["num_inference_steps"]
    cfg = ji.get("guidance_scale") if "guidance_scale" in ef else preset["guidance_scale"]
    shift = ji.get("shift") if "shift" in ef else preset["shift"]
    
    seed = ji.get("seed", -1)
    if seed < 0: seed = int(time.time() * 1000) % (2**31)
    
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    applied = _set_loras(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
    
    start = time.time()
    with torch.inference_mode():
        res = pipeline(prompt=ji["prompt"], width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator(DEVICE).manual_seed(seed), num_images_per_prompt=ji.get("num_images", 1))
    
    final = []
    for img in res.images:
        if ji.get("enable_2nd_pass"):
            _set_loras(pipeline, lora_adapters_loaded, multiplier=ji.get("second_pass_lora_scale_multiplier", 1.0))
            with torch.inference_mode():
                refined = pipeline(prompt=ji["prompt"], image=img, num_inference_steps=ji.get("second_pass_steps", 12), guidance_scale=ji.get("second_pass_guidance_scale", 1.0)).images[0]
            # High-pass merge for style preservation
            b_np = np.asarray(img.convert("RGB"), dtype=np.float32)
            r_rgb = refined.convert("RGB")
            r_low = np.asarray(r_rgb.filter(ImageFilter.GaussianBlur(1.25)), dtype=np.float32)
            diff = (np.asarray(r_rgb, dtype=np.float32) - r_low) * ji.get("second_pass_strength", 0.2)
            img = Image.fromarray(np.clip(b_np + diff, 0, 255).astype(np.uint8))
        final.append(img)

    fmt = ji.get("output_format", "jpeg")
    rt = ji.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    meta = {"loras": applied, "seed": seed, "time": f"{time.time()-start:.2f}s"}
    
    if rt == "s3":
        urls = [upload_to_s3(i, fmt) or encode_base64(i, fmt) for i in final]
        return {"image_urls": urls, "metadata": meta}
    return {"images": [encode_base64(i, fmt) for i in final], "metadata": meta}

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
