"""
RunPod Serverless Handler for FLUX.2-klein-base-9B with LoRA Support
General-purpose multi-LoRA implementation.
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
upscaler_model = None

PRESETS = {
    "realistic_character": {"num_inference_steps": 35, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1024},
    "portrait_hd": {"num_inference_steps": 40, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1536},
    "character_portrait_best": {"num_inference_steps": 45, "guidance_scale": 2.2, "shift": 2.5, "width": 1024, "height": 1024},
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
    "enable_upscale": {"type": bool, "required": False, "default": False},
    "upscale_factor": {"type": float, "required": False, "default": 2.0},
    "upscale_blend": {"type": float, "required": False, "default": 0.35},
}

def _preprocess_job_input(ji):
    if not isinstance(ji, dict): return ji
    ji = dict(ji)
    for k in ["lora_scale", "additional_lora_strength", "additional_lora_scale"]:
        if k in ji and isinstance(ji[k], str):
            try: ji[k] = float(ji[k].strip())
            except: pass
    if isinstance(ji.get("loras"), list):
        for item in ji["loras"]:
            if isinstance(item, dict):
                for sk in ("scale", "strength", "weight"):
                    if sk in item and isinstance(item[sk], str):
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
    image.save(buf, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _set_loras(pipe, adapters, mode="absolute"):
    if not adapters: return []
    applied = []
    scales = [float(a["scale"]) for a in adapters]
    if mode == "normalized" and sum(scales) > 0:
        total = sum(scales)
        scales = [s / total for s in scales]
    
    names = [a["adapter_name"] for a in adapters]
    for i, (name, scale) in enumerate(zip(names, scales)):
        applied.append({"adapter_name": name, "effective_scale": scale, "path": adapters[i]["path"]})
        
    for comp_name in ["transformer", "text_encoder", "text_encoder_2"]:
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "set_adapters"):
            try:
                comp.set_adapters(names, adapter_weights=scales)
                # Manual forcing of scales into PEFT internal scaling dicts
                for m in comp.modules():
                    if hasattr(m, "scaling") and hasattr(m, "adapter_name") and m.adapter_name in names:
                        m.scaling[m.adapter_name] = scales[names.index(m.adapter_name)]
            except: pass
    return applied

def initialize_pipeline(model_id, adapters=None):
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Stable FP8 precision for 24GB+ GPUs
    use_quant = vram < 40.0
    print(f"Loading {model_id} (VRAM: {vram:.1f}GB, Quant: {use_quant})")
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=HF_TOKEN or None)

    if use_quant:
        try:
            import optimum.quanto
            from optimum.quanto import freeze, quantize
            # Standard safe types for optimum-quanto
            qtype = getattr(optimum.quanto, "qfloat8", torch.float8_e4m3fn)
            print(f"Quantizing to {qtype}...")
            quantize(pipe.transformer, weights=qtype)
            freeze(pipe.transformer)
        except Exception as e:
            print(f"Quantization fallback to BF16: {e}")

    loaded = []
    for l in (adapters or []):
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
        import gc; pipeline = None; lora_adapters_loaded = []; gc.collect(); torch.cuda.empty_cache()
        pipeline, lora_adapters_loaded = initialize_pipeline(ji.get("model_id", DEFAULT_MODEL_ID), req_loras)

    preset = PRESETS.get(ji.get("preset"), PRESETS["realistic_character"])
    w, h, steps, cfg, shift = ji.get("width", preset["width"]), ji.get("height", preset["height"]), ji.get("num_inference_steps", preset["num_inference_steps"]), ji.get("guidance_scale", preset["guidance_scale"]), ji.get("shift", preset["shift"])
    
    seed = ji.get("seed", -1)
    if seed < 0: seed = int(time.time() * 1000) % (2**31)
    
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    applied = _set_loras(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
    
    start = time.time()
    with torch.inference_mode():
        res = pipeline(prompt=ji["prompt"], width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator(DEVICE).manual_seed(seed), num_images_per_prompt=ji.get("num_images", 1))
    
    final = []
    for img in res.images:
        # Detailer and Upscaler Logic
        if ji.get("enable_2nd_pass"):
            _set_loras(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
            with torch.inference_mode():
                refined = pipeline(prompt=ji["prompt"], image=img, num_inference_steps=12, guidance_scale=1.0).images[0]
            b_np, r_rgb = np.asarray(img.convert("RGB"), dtype=np.float32), refined.convert("RGB")
            r_low = np.asarray(r_rgb.filter(ImageFilter.GaussianBlur(1.25)), dtype=np.float32)
            diff = (np.asarray(r_rgb, dtype=np.float32) - r_low) * ji.get("second_pass_strength", 0.2)
            img = Image.fromarray(np.clip(b_np + diff, 0, 255).astype(np.uint8))
        if ji.get("enable_upscale"):
            from spandrel import ModelLoader
            m_path = "/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth"
            if os.path.exists(m_path):
                global upscaler_model
                if not upscaler_model: upscaler_model = ModelLoader().load_from_file(m_path).cuda().eval()
                img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
                t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                with torch.inference_mode(): out = upscaler_model(t).cpu().clamp(0, 1)
                img = Image.fromarray((out.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                img = img.resize((int(w * ji.get("upscale_factor", 2.0)), int(h * ji.get("upscale_factor", 2.0))), Image.LANCZOS)
        final.append(img)

    fmt = ji.get("output_format", "jpeg")
    rt = ji.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    meta = {"loras": applied, "seed": seed, "time": f"{time.time()-start:.2f}s"}
    
    if rt == "s3":
        urls = []
        for i in final:
            u = upload_to_s3(i, fmt)
            if u: urls.append(u)
            else:
                b64 = encode_base64(i, fmt)
                if len(b64) > 1_800_000: return {"error": "S3 Upload Failed. Base64 too large for RunPod API."}
                urls.append(b64)
        return {"image_urls": urls, "metadata": meta}
    return {"images": [encode_base64(i, fmt) for i in final], "metadata": meta}

def handler(job):
    try:
        raw = job.get("input", {})
        ji = _preprocess_job_input(raw)
        v = validate(ji, INPUT_SCHEMA)
        if "errors" in v: return {"error": str(v["errors"])}
        return generate_images(v["validated_input"], set(raw.keys()))
    except Exception as e:
        import traceback
        err_msg = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR: {err_msg}")
        return {"error": err_msg, "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
