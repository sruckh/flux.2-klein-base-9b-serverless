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
from diffusers import Flux2KleinPipeline, AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from PIL import Image, ImageFilter

import runpod
from runpod.serverless.utils.rp_validator import validate

# ============================================================================
# Configuration & Constants
# ============================================================================

from huggingface_hub import hf_hub_download
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.2-klein-9B")
DEVICE = os.getenv("DEVICE", "cuda")
TEXT_ENCODER_MODE = os.getenv("TEXT_ENCODER_MODE", "official").strip().lower()

# Abliterated text encoder downloaded as a single safetensors file.
# Transformer, VAE, scheduler, and tokenizer loaded via from_pretrained (uses HF_HOME cache).
# NOTE: The FLUX.2-klein-9b-fp8 single-file uses BFL's custom static-quantized format
# (separate input_scale/weight_scale tensors per layer) which is incompatible with
# FluxTransformer2DModel.load_state_dict — load bf16 transformer from the main repo instead.
MODEL_URLS = {
    "text_encoder": "https://huggingface.co/edicamargo/qwen_3_8b_fp8mixed_abliterated/resolve/main/qwen_3_8b_fp8mixed_abliterated.safetensors",
}

MODEL_BASE_DIR = "/runpod-volume/models/flux2-klein"
UPSCALER_MODEL_PATH = "/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth"

# Globals
pipeline = None
lora_adapters_loaded = []
_last_lora_mode = "absolute"
upscaler_model = None
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_MAX_RETRIES = 3

# ============================================================================
# Schema & Presets
# ============================================================================

# FLUX.2-klein-9B is step+guidance distilled: optimal 4-16 steps, guidance=1.0
PRESETS = {
    "realistic_character":         {"num_inference_steps": 8,  "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1024},
    "portrait_hd":                 {"num_inference_steps": 8,  "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1536},
    "cinematic_full":              {"num_inference_steps": 8,  "guidance_scale": 1.0, "shift": 1.5, "width": 1536, "height": 1024},
    "fast_preview":                {"num_inference_steps": 4,  "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1024},
    "maximum_quality":             {"num_inference_steps": 16, "guidance_scale": 1.0, "shift": 1.0, "width": 1024, "height": 1024},
    "character_portrait_best":     {"num_inference_steps": 12, "guidance_scale": 1.0, "shift": 2.5, "width": 1024, "height": 1024},
    "character_portrait_vertical": {"num_inference_steps": 12, "guidance_scale": 1.0, "shift": 2.0, "width": 896,  "height": 1152},
    "character_cinematic":         {"num_inference_steps": 8,  "guidance_scale": 1.0, "shift": 2.5, "width": 1344, "height": 896},
    "manga_style":                 {"num_inference_steps": 8,  "guidance_scale": 1.0, "shift": 1.5, "width": 1024, "height": 1024},
}

INPUT_SCHEMA = {
    "prompt": {"type": str, "required": True},
    "preset": {"type": str, "required": False, "default": "realistic_character"},
    "width": {"type": int, "required": False, "default": 1024},
    "height": {"type": int, "required": False, "default": 1024},
    "num_inference_steps": {"type": int, "required": False, "default": 8},
    "guidance_scale": {"type": float, "required": False, "default": 1.0},
    "seed": {"type": int, "required": False, "default": -1},
    "num_images": {"type": int, "required": False, "default": 1},
    "output_format": {"type": str, "required": False, "default": "jpeg"},
    "return_type": {"type": str, "required": False, "default": "s3"},
    "shift": {"type": float, "required": False, "default": 1.5},
    "max_sequence_length": {"type": int, "required": False, "default": 512},
    "loras": {"type": list, "required": False, "default": []},
    "lora_scale_mode": {"type": str, "required": False, "default": "absolute"},
    "enable_2nd_pass": {"type": bool, "required": False, "default": False},
    "second_pass_strength": {"type": float, "required": False, "default": 0.2},
    "second_pass_steps": {"type": int, "required": False, "default": 4},
    "second_pass_guidance_scale": {"type": float, "required": False, "default": 1.0},
    "second_pass_lora_scale_multiplier": {"type": float, "required": False, "default": 1.0},
    "enable_upscale": {"type": bool, "required": False, "default": False},
    "upscale_factor": {"type": float, "required": False, "default": 2.0},
    "upscale_blend": {"type": float, "required": False, "default": 0.35},
    "lora_path": {"type": str, "required": False, "default": ""},
    "lora_url": {"type": str, "required": False, "default": ""},
    "lora_scale": {"type": float, "required": False, "default": 0.85},
    "lora_weight_name": {"type": str, "required": False, "default": ""},
    "additional_lora": {"type": str, "required": False, "default": ""},
    "additional_lora_path": {"type": str, "required": False, "default": ""},
    "additional_lora_url": {"type": str, "required": False, "default": ""},
    "additional_lora_scale": {"type": float, "required": False, "default": 0.85},
    "additional_lora_strength": {"type": float, "required": False, "default": 0.85},
    "additional_lora_weight_name": {"type": str, "required": False, "default": ""},
    "addition_lora": {"type": str, "required": False, "default": ""},
    "addition_lora_url": {"type": str, "required": False, "default": ""},
    "addition_lora_scale": {"type": float, "required": False, "default": 0.85},
    "addition_lora_strength": {"type": float, "required": False, "default": 0.85},
    "addition_lora_weight_name": {"type": str, "required": False, "default": ""},
}

# ============================================================================
# Helper Functions (Ordered for Scoping)
# ============================================================================

def _preprocess_job_input(ji: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ji, dict): return ji
    ji = dict(ji)
    fkeys = ["lora_scale", "additional_lora_scale", "additional_lora_strength",
              "addition_lora_scale", "addition_lora_strength", "second_pass_lora_scale_multiplier",
              "second_pass_strength", "upscale_factor", "upscale_blend", "guidance_scale", "shift"]
    for k in fkeys:
        if k in ji and isinstance(ji[k], (str, int, float)):
            try: ji[k] = float(str(ji[k]).strip())
            except: pass
    if isinstance(ji.get("loras"), list):
        for item in ji["loras"]:
            if isinstance(item, dict):
                for sk in ("scale", "strength", "weight", "lora_scale"):
                    if sk in item and isinstance(item[sk], (str, int, float)):
                        try: item[sk] = float(str(item[sk]).strip())
                        except: pass
    return ji

def ensure_models():
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    paths = {}
    for key, url in MODEL_URLS.items():
        after_hf = url.replace("https://huggingface.co/", "")
        repo_id, file_path = after_hf.split("/resolve/main/", 1)
        parts = file_path.rsplit("/", 1)
        subfolder, filename = (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0])
        print(f"Ensuring {key}...")
        paths[key] = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=MODEL_BASE_DIR,
            token=HF_TOKEN or None,
        )
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

def encode_base64(image, fmt):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _collect_requested_loras(ji):
    adapters = []

    for i, l in enumerate(ji.get("loras", []) or []):
        if not isinstance(l, dict):
            continue
        p = l.get("path") or l.get("url") or l.get("lora_url") or l.get("lora_path")
        if not p:
            continue
        adapters.append({
            "path": p,
            "scale": float(next((l[k] for k in ("scale", "strength", "weight", "lora_scale") if l.get(k) is not None), 0.85)),
            "adapter_name": l.get("adapter_name", f"l_{i}"),
            "weight_name": l.get("weight_name"),
        })

    # Legacy fallbacks are still supported and can be combined with `loras`.
    lp = (ji.get("lora_path") or ji.get("lora_url") or "").strip()
    if lp:
        adapters.append({
            "path": lp,
            "scale": float(ji.get("lora_scale", 0.85)),
            "adapter_name": "legacy_l_0",
            "weight_name": ji.get("lora_weight_name"),
        })

    alp = (ji.get("additional_lora") or ji.get("additional_lora_path") or ji.get("additional_lora_url") or ji.get("addition_lora") or ji.get("addition_lora_url") or "").strip()
    if alp:
        ascl = next((ji[k] for k in ("additional_lora_scale", "additional_lora_strength", "addition_lora_scale", "addition_lora_strength") if k in ji and ji[k] is not None), 0.85)
        adapters.append({
            "path": alp,
            "scale": float(ascl),
            "adapter_name": "legacy_l_1",
            "weight_name": ji.get("additional_lora_weight_name") or ji.get("addition_lora_weight_name"),
        })

    # Ensure deterministic/valid adapter names to avoid accidental replacement.
    seen = set()
    for a in adapters:
        name = str(a.get("adapter_name") or "").strip()
        if not name:
            raise ValueError(f"Invalid empty adapter_name for path: {a.get('path')}")
        if name in seen:
            raise ValueError(f"Duplicate adapter_name '{name}'. Adapter names must be unique for multi-LoRA.")
        seen.add(name)

    return adapters

def _download_file(url, dest_path, chunk_size=DOWNLOAD_CHUNK_SIZE, retries=DOWNLOAD_MAX_RETRIES):
    last_error = None

    for attempt in range(1, retries + 1):
        tmp_path = f"{dest_path}.part"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "flux.2-klein-serverless/1.0",
                    "Accept": "*/*",
                },
            )
            with urllib.request.urlopen(req, timeout=300) as response, open(tmp_path, "wb") as out:
                expected_length = response.headers.get("Content-Length")
                expected_length = int(expected_length) if expected_length else None
                downloaded = 0

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)

                out.flush()

            if expected_length is not None and downloaded != expected_length:
                raise IOError(
                    f"retrieval incomplete: got only {downloaded} out of {expected_length} bytes"
                )

            os.replace(tmp_path, dest_path)
            return dest_path
        except Exception as exc:
            last_error = exc
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            if attempt < retries:
                sleep_seconds = min(2 ** (attempt - 1), 8)
                print(
                    f"LoRA download retry {attempt}/{retries} failed for {url}: {exc}. "
                    f"Retrying in {sleep_seconds}s..."
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download LoRA from '{url}' after {retries} attempts: {last_error}")

def _classify_adapter_checkpoint(adapter_path):
    try:
        with safe_open(adapter_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return None

    if not keys:
        return None

    lowered = [k.lower() for k in keys]
    if any(".lokr_" in k or k.endswith("lokr_w1") or k.endswith("lokr_w2") for k in lowered):
        return "lokr"
    if any(".hada_" in k or ".hada_w" in k for k in lowered):
        return "hada"
    if any(".lora_" in k or ".dora_" in k or ".lora_up." in k or ".lora_down." in k for k in lowered):
        return "lora"
    if any(".locon_" in k for k in lowered):
        return "locon"
    return "unknown"

def _resolve_adapter_file(path, weight_name=None):
    if os.path.isfile(path):
        return path
    if weight_name and os.path.isdir(path):
        candidate = os.path.join(path, weight_name)
        if os.path.isfile(candidate):
            return candidate
    return None

def _validate_adapter_checkpoint(adapter_path, adapter_name, source_path):
    adapter_type = _classify_adapter_checkpoint(adapter_path)
    if adapter_type in {"lokr", "hada"}:
        raise ValueError(
            f"Adapter '{adapter_name}' from '{source_path}' is a LyCORIS/{adapter_type.upper()} checkpoint, "
            "which this server cannot load with diffusers `load_lora_weights()`. "
            "Use a standard FLUX LoRA/LoCon safetensors file instead."
        )

def _sanitize_adapter_checkpoint_for_diffusers(adapter_path):
    try:
        with safe_open(adapter_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        return None

    if not keys:
        return None

    non_lora_keys = [k for k in keys if "lora" not in k.lower()]
    # OneTrainer FLUX LoRAs include valid `.alpha` tensors alongside
    # `lora_down`/`lora_up` weights, but some diffusers loader paths reject
    # checkpoints unless every key contains `lora`. Strip only those alpha
    # tensors for compatibility; keep the original file untouched.
    if not non_lora_keys or any(not k.endswith(".alpha") for k in non_lora_keys):
        return None

    # OneTrainer "diffusers dot-notation" format: keys start with `transformer.`
    # and use `lora_down`/`lora_up` instead of PEFT's `lora_A`/`lora_B`.
    # diffusers `load_lora_weights` only recognises kohya (`lora_unet_*`) and
    # PEFT (`.lora_A.`) formats; encountering `transformer.*.lora_down.*` keys
    # causes it to misroute to the kohya converter which raises
    # `list index out of range` while parsing the incompatible key structure.
    # Rename to PEFT convention so diffusers loads the weights directly.
    is_onetrainer_dot_format = (
        any(k.startswith("transformer.") for k in keys)
        and any(".lora_down." in k for k in keys)
        and not any(".lora_A." in k for k in keys)
    )

    state_dict = load_safetensors(adapter_path)
    filtered = {k: v for k, v in state_dict.items() if not k.endswith(".alpha")}
    alpha_removed = len(state_dict) - len(filtered)

    if is_onetrainer_dot_format:
        filtered = {
            k.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B."): v
            for k, v in filtered.items()
        }
        print(
            f"Sanitized LoRA checkpoint for diffusers compatibility: "
            f"removed {alpha_removed} alpha tensors, "
            f"renamed lora_down/lora_up to lora_A/lora_B (OneTrainer dot format)"
        )
    else:
        print(
            f"Sanitized LoRA checkpoint for diffusers compatibility: removed {alpha_removed} alpha tensors"
        )
    return {"dir": sanitized_dir, "weight_name": sanitized_name}

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
    # Flux2LoraLoaderMixin defines LoRA-compatible modules and routes adapter weights internally.
    pipe.set_adapters(names, adapter_weights=scales)
    return applied

def initialize_pipeline(adapters=None, scale_mode="absolute"):
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Loading Components (VRAM: {vram_gb:.1f}GB)...")

    if TEXT_ENCODER_MODE not in {"official", "abliterated"}:
        raise ValueError(
            f"Unsupported TEXT_ENCODER_MODE='{TEXT_ENCODER_MODE}'. Use 'official' or 'abliterated'."
        )

    # Load the full pipeline so diffusers resolves the correct transformer class
    # for FLUX.2-klein-9B's architecture (Flux2Transformer2DModel).
    print("Loading Flux2KleinPipeline from pretrained...")
    pipe = Flux2KleinPipeline.from_pretrained(
        DEFAULT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN or None,
    )

    if TEXT_ENCODER_MODE == "abliterated":
        # Avoid creating meta modules: load custom weights into the instantiated pipeline encoder.
        paths = ensure_models()
        print("Applying abliterated text encoder weights...")
        te_sd = load_safetensors(paths["text_encoder"])
        pipe.text_encoder.load_state_dict(te_sd, strict=True, assign=True)
        pipe.text_encoder.eval()

    loaded_adapters = []
    if adapters:
        for l in adapters:
            try:
                if l['path'].startswith("http"):
                    with tempfile.TemporaryDirectory() as tmp:
                        p = os.path.join(tmp, "lora.safetensors")
                        _download_file(l["path"], p)
                        _validate_adapter_checkpoint(p, l["adapter_name"], l["path"])
                        sanitized = _sanitize_adapter_checkpoint_for_diffusers(p)
                        if sanitized is None:
                            pipe.load_lora_weights(tmp, weight_name="lora.safetensors", adapter_name=l['adapter_name'])
                        else:
                            pipe.load_lora_weights(
                                sanitized["dir"],
                                weight_name=sanitized["weight_name"],
                                adapter_name=l['adapter_name'],
                            )
                else:
                    load_kwargs = {"adapter_name": l["adapter_name"]}
                    if l.get("weight_name"):
                        load_kwargs["weight_name"] = l["weight_name"]
                    adapter_file = _resolve_adapter_file(l["path"], l.get("weight_name"))
                    if adapter_file:
                        _validate_adapter_checkpoint(adapter_file, l["adapter_name"], l["path"])
                        sanitized = _sanitize_adapter_checkpoint_for_diffusers(adapter_file)
                        if sanitized is not None:
                            pipe.load_lora_weights(
                                sanitized["dir"],
                                weight_name=sanitized["weight_name"],
                                adapter_name=l["adapter_name"],
                            )
                            loaded_adapters.append(l)
                            continue
                    pipe.load_lora_weights(l['path'], **load_kwargs)
                loaded_adapters.append(l)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load LoRA adapter '{l['adapter_name']}' from '{l['path']}': {e}"
                ) from e

    if loaded_adapters:
        names = [a["adapter_name"] for a in loaded_adapters]
        scales = [float(a["scale"]) for a in loaded_adapters]
        if scale_mode == "normalized" and sum(scales) > 0:
            total = sum(scales)
            scales = [s / total for s in scales]
        pipe.set_adapters(names, adapter_weights=scales)
        pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
        pipe.unload_lora_weights()

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe, loaded_adapters

def _detail_only_blend(base_img, refined_img, strength):
    # Keep color/composition from base image and inject only luminance micro-detail.
    strength = float(np.clip(strength, 0.0, 0.35))
    base_ycc = np.asarray(base_img.convert("YCbCr"), dtype=np.float32)
    refined_y = np.asarray(refined_img.convert("YCbCr"), dtype=np.float32)[..., 0]
    base_y = base_ycc[..., 0]

    refined_low = np.asarray(
        Image.fromarray(refined_y.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.5)),
        dtype=np.float32,
    )
    detail = refined_y - refined_low

    base_low = np.asarray(
        Image.fromarray(base_y.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(1.2)),
        dtype=np.float32,
    )
    edge = np.abs(base_y - base_low)
    edge = edge / (edge.max() + 1e-6)
    edge = np.clip((edge - 0.05) / 0.95, 0.0, 1.0)

    base_ycc[..., 0] = np.clip(base_y + detail * edge * strength, 0.0, 255.0)
    return Image.fromarray(base_ycc.astype(np.uint8), mode="YCbCr").convert("RGB")

def tiled_upscale(image, factor, blend):
    from spandrel import ModelLoader
    global upscaler_model
    m_path = UPSCALER_MODEL_PATH
    if not os.path.exists(m_path):
        os.makedirs(os.path.dirname(m_path), exist_ok=True)
        urllib.request.urlretrieve("https://github.com/Phhofm/models/releases/download/4xRealWebPhoto_v4_drct-l/4xRealWebPhoto_v4_drct-l.pth", m_path)
    if not upscaler_model: upscaler_model = ModelLoader().load_from_file(m_path).cuda().eval()

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

def generate_images(ji):
    global pipeline, lora_adapters_loaded, _last_lora_mode
    req_loras = _collect_requested_loras(ji)
    mode = ji.get("lora_scale_mode", "absolute")

    sig = lambda x, m: tuple((y["path"], y["adapter_name"], round(float(y["scale"]), 4)) for y in x) + (m,)
    if pipeline is None or sig(req_loras, mode) != sig(lora_adapters_loaded, _last_lora_mode):
        print("Resetting VRAM for model change...")
        pipeline = None
        import gc
        for _ in range(3): gc.collect(); torch.cuda.empty_cache()
        pipeline, lora_adapters_loaded = initialize_pipeline(req_loras, scale_mode=mode)
        _last_lora_mode = mode

    preset = PRESETS.get(ji.get("preset"), PRESETS["realistic_character"]).copy()
    w, h = ji.get("width") or preset["width"], ji.get("height") or preset["height"]
    steps = ji.get("num_inference_steps") or preset["num_inference_steps"]
    requested_cfg = ji.get("guidance_scale") or preset["guidance_scale"]
    cfg = requested_cfg
    if getattr(getattr(pipeline, "config", None), "is_distilled", False) and cfg > 1.0:
        print(f"Distilled model active; overriding guidance_scale {cfg} -> 1.0")
        cfg = 1.0
    shift = ji.get("shift") or preset["shift"]
    max_seq_len = ji.get("max_sequence_length", 512)

    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config, shift=shift)
    applied = [{"adapter_name": a["adapter_name"], "effective_scale": float(a["scale"]), "path": a["path"]} for a in lora_adapters_loaded]

    print(f"Inference: {w}x{h}, {steps} steps, CFG {cfg}")
    start_time = time.time()
    seed = ji.get("seed", -1)
    if seed < 0:
        seed = int(time.time())
    with torch.no_grad():
        res = pipeline(
            prompt=ji["prompt"], width=w, height=h,
            num_inference_steps=steps, guidance_scale=cfg,
            generator=torch.Generator(DEVICE).manual_seed(seed),
            num_images_per_prompt=ji.get("num_images", 1),
            max_sequence_length=max_seq_len,
        )

    final = []
    for img in res.images:
        # STEP 1: SR upscale first (before second pass)
        if ji.get("enable_upscale"):
            img = tiled_upscale(img, ji.get("upscale_factor", 2.0), ji.get("upscale_blend", 0.35))

        # STEP 2: FLUX second pass at (potentially upscaled) resolution
        if ji.get("enable_2nd_pass"):
            requested_second_pass_steps = int(ji.get("second_pass_steps", 4))
            second_pass_steps = max(1, min(requested_second_pass_steps, 8))
            if second_pass_steps != requested_second_pass_steps:
                print(f"Detail-only second pass: clamping second_pass_steps {requested_second_pass_steps} -> {second_pass_steps}")

            requested_second_pass_cfg = float(ji.get("second_pass_guidance_scale", 1.0))
            second_pass_cfg = 1.0
            if requested_second_pass_cfg != 1.0:
                print(f"Detail-only second pass: overriding second_pass_guidance_scale {requested_second_pass_cfg} -> 1.0")

            with torch.no_grad():
                refined = pipeline(
                    prompt=ji["prompt"],
                    image=img,
                    width=img.width,
                    height=img.height,
                    num_inference_steps=second_pass_steps,
                    guidance_scale=second_pass_cfg,
                    generator=torch.Generator(DEVICE).manual_seed(seed),
                ).images[0]
            img = _detail_only_blend(img.convert("RGB"), refined.convert("RGB"), ji.get("second_pass_strength", 0.2))

        final.append(img)

    fmt = ji.get("output_format", "jpeg")
    rt = ji.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    meta = {
        "loras": applied,
        "generation_time": f"{time.time()-start_time:.2f}s",
        "requested_guidance_scale": requested_cfg,
        "effective_guidance_scale": cfg,
    }

    if rt == "s3":
        urls = []
        for i in final:
            u = upload_to_s3(i, fmt)
            if u: urls.append(u)
            else:
                b64 = encode_base64(i, fmt)
                if len(b64) > 1_800_000: return {"error": "S3 Failed and Base64 > 2MB."}
                urls.append(b64)
        return {"image_urls": urls, "metadata": meta}
    return {"images": [encode_base64(i, fmt) for i in final], "metadata": meta}

def handler(job):
    try:
        raw = job.get("input", {})
        ji = _preprocess_job_input(raw)
        val = validate(ji, INPUT_SCHEMA)
        if "errors" in val: return {"error": str(val["errors"])}
        return generate_images(val["validated_input"])
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"CRITICAL ERROR: {e}\n{tb}")
        return {"error": f"{type(e).__name__}: {str(e)}", "traceback": tb}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
