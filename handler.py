# Set VRAM fragmentation protection before ANY imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["QUANTO_DISABLE_MARLIN"] = "1"

import base64
import io
import json
import tempfile
import time
import urllib.request
import uuid
from typing import Dict, Any, List, Optional, Union

import numpy as np
import boto3
import torch
from botocore.exceptions import ClientError
from diffusers import FluxTransformer2DModel, Flux2KleinPipeline, AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image, ImageFilter

import runpod
from runpod.serverless.utils.rp_validator import validate

# Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cuda")

# Component URLs (ComfyUI Style - Pre-quantized FP8)
MODEL_URLS = {
    "transformer": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/resolve/main/flux-2-klein-base-9b-fp8.safetensors",
    "text_encoder": "https://huggingface.co/edicamargo/qwen_3_8b_fp8mixed_abliterated/resolve/main/qwen_3_8b_fp8mixed_abliterated.safetensors",
    "vae": "https://huggingface.co/loljj0001/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors"
}

MODEL_BASE_DIR = "/runpod-volume/models/flux2-klein"
UPSCALER_MODEL_PATH = "/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth"

# Globals
pipeline = None
lora_adapters_loaded = []
upscaler_model = None

PRESETS = {
    "realistic_character": {"num_inference_steps": 35, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1024},
    "portrait_hd": {"num_inference_steps": 40, "guidance_scale": 2.0, "shift": 1.5, "width": 1024, "height": 1536},
    "manga_style": {"num_inference_steps": 25, "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1024},
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
    "max_sequence_length": {"type": int, "required": False, "default": 512},
    "enable_2nd_pass": {"type": bool, "required": False, "default": False},
    "second_pass_strength": {"type": float, "required": False, "default": 0.2},
    "enable_upscale": {"type": bool, "required": False, "default": False},
    "upscale_factor": {"type": float, "required": False, "default": 2.0},
    "upscale_blend": {"type": float, "required": False, "default": 0.35},
    "lora_path": {"type": str, "required": False, "default": ""},
    "lora_scale": {"type": float, "required": False, "default": 0.85},
}

def ensure_models():
    """Check if components exist in cache, otherwise download."""
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    
    paths = {}
    for key, url in MODEL_URLS.items():
        filename = url.split("/")[-1]
        local_path = os.path.join(MODEL_BASE_DIR, filename)
        paths[key] = local_path
        if not os.path.exists(local_path):
            print(f"Downloading {key} component to {local_path}...")
            urllib.request.urlretrieve(url, local_path)
    return paths

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

def _set_loras(pipe, adapters, mode="absolute"):
    if not adapters: return []
    scales = [float(a["scale"]) for a in adapters]
    names = [a["adapter_name"] for a in adapters]
    applied = []
    for i, (name, scale) in enumerate(zip(names, scales)):
        applied.append({"adapter_name": name, "effective_scale": scale})
        
    for comp_name in ["transformer", "text_encoder"]:
        comp = getattr(pipe, comp_name, None)
        if comp is not None and hasattr(comp, "set_adapters"):
            try:
                comp.set_adapters(names, adapter_weights=scales)
                for m in comp.modules():
                    if hasattr(m, "scaling") and hasattr(m, "adapter_name") and m.adapter_name in names:
                        m.scaling[m.adapter_name] = scales[names.index(m.adapter_name)]
            except: pass
    return applied

def initialize_pipeline(adapters=None):
    print("Initializing FLUX.2-klein using ComfyUI-style component loading...")
    paths = ensure_models()
    
    # 1. Load VAE
    vae = AutoencoderKL.from_single_file(paths["vae"], torch_dtype=torch.bfloat16)
    
    # 2. Load Transformer (Pre-quantized FP8)
    transformer = FluxTransformer2DModel.from_single_file(paths["transformer"], torch_dtype=torch.float8_e4m3fn)
    
    # 3. Load Qwen Text Encoder (FP8 Mixed)
    # Klein uses Qwen-8B. We load it as an AutoModelForCausalLM and extract the encoder part if possible, 
    # or use the community diffusers-compatible Qwen loader.
    text_encoder = AutoModelForCausalLM.from_single_file(paths["text_encoder"], torch_dtype=torch.float8_e4m3fn)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct") # Klein uses Qwen tokenizer
    
    # 4. Construct Pipeline
    pipe = Flux2KleinPipeline(
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained("black-forest-labs/FLUX.2-klein-base-9B", subfolder="scheduler")
    )

    if adapters:
        for l in adapters:
            try:
                if l['path'].startswith("http"):
                    with tempfile.TemporaryDirectory() as tmp:
                        p = os.path.join(tmp, "lora.safetensors")
                        urllib.request.urlretrieve(l['path'], p)
                        pipe.load_lora_weights(tmp, weight_name="lora.safetensors", adapter_name=l['adapter_name'])
                else: pipe.load_lora_weights(l['path'], adapter_name=l['adapter_name'])
            except Exception as e: print(f"LoRA Error: {e}")
    
    _set_loras(pipe, adapters)
    
    # COMPONENT-BASED OFFLOAD: Frees ~16GB text encoder immediately
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()
    return pipe, adapters

def tiled_upscale(image, factor, blend):
    from spandrel import ModelLoader
    global upscaler_model
    if not os.path.exists(UPSCALER_MODEL_PATH):
        os.makedirs(os.path.dirname(UPSCALER_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URLS["upscaler"], UPSCALER_MODEL_PATH)
    if not upscaler_model: upscaler_model = ModelLoader().load_from_file(UPSCALER_MODEL_PATH).cuda().eval()
    
    scale = upscaler_model.scale
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    in_h, in_w = tensor.shape[2], tensor.shape[3]
    out_h, out_w = in_h * scale, in_w * scale
    output, weight = torch.zeros((1, 3, out_h, out_w)), torch.zeros((1, 1, out_h, out_w))
    tile_size, overlap = 384, 64
    step = tile_size - overlap
    
    y_starts = list(range(0, max(1, in_h - tile_size), step)) + [max(0, in_h - tile_size)]
    x_starts = list(range(0, max(1, in_w - tile_size), step)) + [max(0, in_w - tile_size)]
    
    for y in sorted(set(y_starts)):
        for x in sorted(set(x_starts)):
            tile = tensor[:, :, y:y+tile_size, x:x+tile_size].to(DEVICE)
            with torch.inference_mode(): tile_out = upscaler_model(tile).cpu().clamp(0.0, 1.0)
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

def generate_images(ji, ef):
    global pipeline, lora_adapters_loaded
    req_loras = []
    if "loras" in ef:
        for i, l in enumerate(ji.get("loras", [])):
            p = l.get("path") or l.get("url") or l.get("lora_url") or l.get("lora_path")
            if p: req_loras.append({"path": p, "scale": float(l.get("scale", 0.85)), "adapter_name": l.get("adapter_name", f"l_{i}")})
    else:
        lp = (ji.get("lora_path") or ji.get("lora_url") or "").strip()
        if lp: req_loras.append({"path": lp, "scale": float(ji.get("lora_scale", 0.85)), "adapter_name": "l_0"})

    sig = lambda x: tuple((y["path"], y["adapter_name"]) for y in x)
    if pipeline is None or sig(req_loras) != sig(lora_adapters_loaded):
        pipeline = None
        import gc
        for _ in range(3): gc.collect(); torch.cuda.empty_cache()
        pipeline, lora_adapters_loaded = initialize_pipeline(req_loras)

    preset = PRESETS.get(ji.get("preset"), PRESETS["realistic_character"]).copy()
    w, h = ji.get("width") or preset["width"], ji.get("height") or preset["height"]
    steps, cfg, shift = ji.get("num_inference_steps") or preset["num_inference_steps"], ji.get("guidance_scale") or preset["guidance_scale"], ji.get("shift") or preset["shift"]
    
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    applied = _set_loras(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
    
    print(f"Inference: {w}x{h}, {steps} steps, CFG {cfg}")
    start_time = time.time()
    with torch.inference_mode():
        res = pipeline(prompt=ji["prompt"], width=w, height=h, num_inference_steps=steps, guidance_scale=cfg, generator=torch.Generator(DEVICE).manual_seed(ji.get("seed", -1) if ji.get("seed", -1) >= 0 else int(time.time())), num_images_per_prompt=ji.get("num_images", 1), max_sequence_length=ji.get("max_sequence_length", 512))
    
    final = []
    for img in res.images:
        if ji.get("enable_2nd_pass"):
            _set_loras(pipeline, lora_adapters_loaded, mode=ji.get("lora_scale_mode", "absolute"))
            with torch.inference_mode():
                refined = pipeline(prompt=ji["prompt"], image=img, num_inference_steps=12, guidance_scale=1.0).images[0]
            b_np, r_rgb = np.asarray(img.convert("RGB"), dtype=np.float32), refined.convert("RGB")
            r_low = np.asarray(r_rgb.filter(ImageFilter.GaussianBlur(1.25)), dtype=np.float32)
            diff = (np.asarray(r_rgb, dtype=np.float32) - r_low) * ji.get("second_pass_strength", 0.2)
            img = Image.fromarray(np.clip(b_np + diff, 0, 255).astype(np.uint8))
        if ji.get("enable_upscale"):
            img = tiled_upscale(img, ji.get("upscale_factor", 2.0), ji.get("upscale_blend", 0.35))
        final.append(img)

    fmt = ji.get("output_format", "jpeg")
    rt = ji.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    meta = {"loras": applied, "generation_time": f"{time.time()-start_time:.2f}s"}
    
    if rt == "s3":
        urls = []
        for i in final:
            u = upload_to_s3(i, fmt)
            if u: urls.append(u)
            else: urls.append(base64.b64encode(i.tobytes()).decode())
        return {"image_urls": urls, "metadata": meta}
    return {"images": [encode_base64(i, fmt) for i in final], "metadata": meta}

def handler(job):
    try:
        raw = job.get("input", {})
        ji = _preprocess_job_input(raw)
        val = validate(ji, INPUT_SCHEMA)
        if "errors" in val: return {"error": str(val["errors"])}
        return generate_images(val["validated_input"], set(raw.keys()))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"CRITICAL ERROR: {e}\n{tb}")
        return {"error": f"{type(e).__name__}: {str(e)}", "traceback": tb}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
