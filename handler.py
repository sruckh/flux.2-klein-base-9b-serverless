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
# expandable_segments lets PyTorch reuse fragmented reserved-but-unallocated
# blocks instead of requiring a contiguous free region — prevents OOM on 24 GB
# GPUs where Marlin fp8 packing needs a small extra allocation after loading.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Disable Marlin fp8 kernels — they require contiguous input tensors which
# diffusers does not guarantee during the transformer forward pass, causing
# "RuntimeError: A is not contiguous". Non-Marlin fp8 handles arbitrary layout.
# Use direct assignment (not setdefault) so RunPod env vars cannot override this.
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
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")  # For S3-compatible services
S3_PRESIGNED_URL_EXPIRY = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))  # 1 hour default

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

DEFAULT_MODEL_ID = os.getenv(
    "MODEL_ID",
    "black-forest-labs/FLUX.2-klein-base-9B"
)

DEFAULT_LORA_PATH = os.getenv("DEFAULT_LORA_PATH", "")
DEFAULT_LORA_SCALE = float(os.getenv("DEFAULT_LORA_SCALE", "1.0"))

DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = os.getenv("DTYPE", "float8_e4m3fn").lower()
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "true").lower() == "true"

# When True, use enable_model_cpu_offload() instead of pipeline.to(DEVICE).
# Only needed if total model weight exceeds GPU VRAM (e.g. bf16 on a 32 GB GPU).
# With DTYPE=float8_e4m3fn the transformer is quantized to fp8 (~9 GB), bringing
# total to ~14-16 GB — fits RTX 4090 (24 GB) and 32 GB GPUs without offload.
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "false").lower() == "true"

# Map dtype string to torch dtype
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    # fp8 keeps weights in FP8 format on H100/Ada GPUs (requires CUDA compute ≥ 8.9)
    "float8": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


# ============================================================================
# Upscaler Configuration
# ============================================================================

# DRCT-L 4x super-resolution model trained on real-world web photos.
# Source: https://github.com/Phhofm/models
UPSCALER_MODEL_URL = (
    "https://github.com/Phhofm/models/releases/download/"
    "4xRealWebPhoto_v4_drct-l/4xRealWebPhoto_v4_drct-l.pth"
)
UPSCALER_MODEL_PATH = os.getenv(
    "UPSCALER_MODEL_PATH",
    "/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth",
)
UPSCALE_TILE_SIZE = 512    # input-space tile size in pixels
UPSCALE_TILE_OVERLAP = 32  # input-space overlap between tiles in pixels


# ============================================================================
# Optimized Presets for Different Use Cases
# Based on research from:
# - Black Forest Labs FLUX.2 documentation
# - fal.ai FLUX.2 training guides
# - Community testing and best practices
# ============================================================================

PRESETS = {
    # Primary preset for photorealistic human portraits with character LoRA.
    # FLUX.2-klein-base (undistilled): guidance 2.5 avoids the over-rendered
    # plastic look that higher values produce. shift=2.0 yields natural skin
    # texture and lighting rather than hyper-stylised structure.
    "realistic_character": {
        "num_inference_steps": 35,
        "guidance_scale": 2.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1024,
        "description": "Photorealistic human portrait — square crop"
    },

    # Portrait orientation (2:3) — most natural for head/shoulders shots.
    # Slightly more steps for the extra pixel budget.
    "portrait_hd": {
        "num_inference_steps": 40,
        "guidance_scale": 2.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1536,
        "description": "Photorealistic portrait — 2:3 vertical crop"
    },

    # Full-body or environmental shot in cinematic 3:2 ratio.
    "cinematic_full": {
        "num_inference_steps": 35,
        "guidance_scale": 2.5,
        "shift": 1.5,
        "width": 1536,
        "height": 1024,
        "description": "Cinematic full-body or environment shot — 3:2 horizontal"
    },

    # Fast iteration for prompt / seed testing — not for final output.
    "fast_preview": {
        "num_inference_steps": 20,
        "guidance_scale": 2.0,
        "shift": 1.5,
        "width": 1024,
        "height": 1024,
        "description": "Quick preview for prompt and seed exploration"
    },

    # Highest quality — use for hero shots where generation time is acceptable.
    "maximum_quality": {
        "num_inference_steps": 50,
        "guidance_scale": 2.5,
        "shift": 1.0,
        "width": 1024,
        "height": 1024,
        "description": "Maximum quality — slowest, most detail"
    },

    # =========================================================================
    # Character LoRA Optimized Presets
    # =========================================================================

    # Maximum character fidelity with natural skin — tuned for character LoRAs.
    # Uses BFL-recommended shift=2.5 for character LoRAs, balanced guidance,
    # and extra steps for the most natural, detailed output.
    "character_portrait_best": {
        "num_inference_steps": 45,
        "guidance_scale": 2.2,
        "shift": 2.5,
        "width": 1024,
        "height": 1024,
        "description": "Best quality for character LoRA portraits — natural skin, maximum fidelity"
    },

    # Portrait orientation for head/shoulders — ~4:5 ratio ideal for faces.
    # Lower shift for more organic detail in close-ups.
    "character_portrait_vertical": {
        "num_inference_steps": 45,
        "guidance_scale": 2.0,
        "shift": 2.0,
        "width": 896,
        "height": 1152,
        "description": "Character LoRA — vertical portrait, 4:5 ratio for head/shoulders"
    },

    # Cinematic 3:2 horizontal — for full-body or environmental character shots.
    # Higher guidance for stronger composition coherence.
    "character_cinematic": {
        "num_inference_steps": 40,
        "guidance_scale": 2.5,
        "shift": 2.5,
        "width": 1344,
        "height": 896,
        "description": "Character LoRA — cinematic 3:2 horizontal, full-body or environmental"
    },
}


# ============================================================================
# Global Pipeline Instance
# ============================================================================

pipeline: Optional[Flux2KleinPipeline] = None
model_loaded = False
lora_path_loaded: str = DEFAULT_LORA_PATH
upscaler_model = None  # spandrel ImageModelDescriptor, loaded on first upscale request


# ============================================================================
# Input Validation Schema
# ============================================================================

INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
        "constraints": lambda x: len(x.strip()) > 0,
    },
    "preset": {
        "type": str,
        "required": False,
        "default": "realistic_character",
        "constraints": lambda x: x in PRESETS or x == "",
    },
    "width": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda x: x > 0 and x % 16 == 0,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1024,
        "constraints": lambda x: x > 0 and x % 16 == 0,
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 28,
        "constraints": lambda x: 1 <= x <= 100,
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 4.0,  # BFL default for base (undistilled) model; range 3.5-5.0
        "constraints": lambda x: x >= 0,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": -1,
    },
    "num_images": {
        "type": int,
        "required": False,
        "default": 1,
        "constraints": lambda x: 1 <= x <= 4,
    },
    "lora_path": {
        "type": str,
        "required": False,
        "default": DEFAULT_LORA_PATH,
    },
    "lora_scale": {
        "type": float,
        "required": False,
        "default": DEFAULT_LORA_SCALE,
        "constraints": lambda x: 0.0 <= x <= 2.0,
    },
    "output_format": {
        "type": str,
        "required": False,
        "default": "jpeg",
        "constraints": lambda x: x.lower() in ["png", "jpeg", "webp"],
    },
    "return_type": {
        "type": str,
        "required": False,
        "default": "s3" if S3_BUCKET_NAME else "base64",
        "constraints": lambda x: x.lower() in ["s3", "base64"],
    },
    "max_sequence_length": {
        "type": int,
        "required": False,
        "default": 512,
        "constraints": lambda x: x > 0,
    },
    "shift": {
        "type": float,
        "required": False,
        "default": 3.0,  # BFL recommendation: 3.0 for character LoRAs; tune 1.0-7.0
        "constraints": lambda x: 0.1 <= x <= 10.0,
    },
    # -------------------------------------------------------------------------
    # Prompt Enhancement
    # -------------------------------------------------------------------------
    # NOTE: caption_upsample_temperature is NOT supported by Flux2KleinPipeline.
    # It's only available in Flux2Pipeline (dev variant). Removed to avoid errors.
    "prompt_2": {
        "type": str,
        "required": False,
        # Optional secondary prompt for prompt weighting. When provided with
        # prompt_2_weight, the final prompt becomes a weighted blend:
        #   final = prompt * (1 - weight) + prompt_2 * weight
        # Example: prompt="TOK woman", prompt_2="cinematic lighting", weight=0.3
        "default": "",
    },
    "prompt_2_weight": {
        "type": float,
        "required": False,
        # Weight for prompt_2 in the blend (0.0 = 100% prompt, 1.0 = 100% prompt_2).
        # Recommended range: 0.1–0.5 for subtle style influence.
        "default": 0.0,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    # -------------------------------------------------------------------------
    # 2nd Pass Detailer (low-denoise img2img for fine detail refinement)
    # -------------------------------------------------------------------------
    "enable_2nd_pass": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "second_pass_strength": {
        "type": float,
        "required": False,
        # Blend factor for how much 2nd-pass micro-detail to transfer back
        # onto the 1st pass image. 0.2 is intentionally conservative.
        "default": 0.2,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
    "second_pass_steps": {
        "type": int,
        "required": False,
        "default": 12,
        "constraints": lambda x: 5 <= x <= 50,
    },
    "second_pass_guidance_scale": {
        "type": float,
        "required": False,
        "default": 1.0,  # Much lower than 1st pass — high CFG on a near-clean latent
        # causes over-saturation and plastic/cartoon look. 1.0–1.3 recommended.
        "constraints": lambda x: 0.1 <= x <= 5.0,
    },
    # -------------------------------------------------------------------------
    # Tiled Upscaler (DRCT-L 4x → resize to target factor)
    # -------------------------------------------------------------------------
    "enable_upscale": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "upscale_factor": {
        "type": float,
        "required": False,
        "default": 2.0,  # 2.0 = 4x DRCT then resize to 2x (2048×2048 from 1024×1024)
        "constraints": lambda x: 0.25 <= x <= 4.0,
    },
    "upscale_blend": {
        "type": float,
        "required": False,
        # 0.35 keeps identity/composition from the source while adding detail.
        "default": 0.35,
        "constraints": lambda x: 0.0 <= x <= 1.0,
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_to_base64(image: Image.Image, format: str = "jpeg") -> str:
    """Encode a PIL Image to base64 string."""
    buffer = io.BytesIO()
    format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WebP"}
    # Use JPEG quality optimization for smaller payload
    if format.lower() == "jpeg":
        image.save(buffer, format="JPEG", quality=95, optimize=True)
    else:
        image.save(buffer, format=format_map.get(format.lower(), "PNG"))
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def upload_to_s3(image: Image.Image, format: str = "jpeg") -> Optional[str]:
    """
    Upload image to S3 and return presigned URL.
    
    Args:
        image: PIL Image to upload
        format: Image format (jpeg, png, webp)
    
    Returns:
        Presigned URL for the uploaded image or None if S3 is not configured
    """
    s3_client = get_s3_client()
    if not s3_client:
        return None
    
    # Generate unique filename
    file_extension = format.lower()
    if file_extension == "jpeg":
        file_extension = "jpg"
    filename = f"flux2-klein/{uuid.uuid4()}.{file_extension}"
    
    # Convert image to bytes
    buffer = io.BytesIO()
    if format.lower() == "jpeg":
        image.save(buffer, format="JPEG", quality=95, optimize=True)
        content_type = "image/jpeg"
    elif format.lower() == "webp":
        image.save(buffer, format="WebP", quality=95)
        content_type = "image/webp"
    else:
        image.save(buffer, format="PNG")
        content_type = "image/png"
    
    buffer.seek(0)
    
    try:
        # Upload to S3
        s3_client.upload_fileobj(
            buffer,
            S3_BUCKET_NAME,
            filename,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "max-age=31536000",  # 1 year cache
            }
        )
        
        # Generate presigned URL
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": filename},
            ExpiresIn=S3_PRESIGNED_URL_EXPIRY,
        )
        
        return url
    
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None


def download_upscaler_model() -> str:
    """
    Ensure the DRCT-L upscaler .pth is present on the network volume.
    Downloads from GitHub releases if not already cached.

    Returns:
        Absolute path to the cached model file.
    """
    model_path = UPSCALER_MODEL_PATH
    if os.path.exists(model_path):
        print(f"Upscaler model already cached: {model_path}")
        return model_path

    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)
    print(f"Downloading upscaler model from:\n  {UPSCALER_MODEL_URL}")
    try:
        urllib.request.urlretrieve(UPSCALER_MODEL_URL, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Upscaler model downloaded ({size_mb:.1f} MB) → {model_path}")
    except Exception as e:
        # Clean up partial download
        if os.path.exists(model_path):
            os.remove(model_path)
        raise RuntimeError(f"Failed to download upscaler model: {e}") from e
    return model_path


def load_upscaler():
    """
    Load and cache the DRCT-L upscaler model via spandrel (lazy, one-time).

    Returns:
        spandrel ImageModelDescriptor in CUDA eval mode.
    """
    global upscaler_model
    if upscaler_model is not None:
        return upscaler_model

    from spandrel import ImageModelDescriptor, ModelLoader

    model_path = download_upscaler_model()
    print(f"Loading upscaler model from {model_path} ...")
    model = ModelLoader().load_from_file(model_path)
    if not isinstance(model, ImageModelDescriptor):
        raise RuntimeError(
            f"Upscaler model is not an ImageModelDescriptor (got {type(model).__name__}). "
            "Check that the .pth file is a valid super-resolution model."
        )
    model = model.cuda().eval()
    upscaler_model = model
    print(f"Upscaler loaded: {model.architecture.name}, scale={model.scale}x")
    return upscaler_model


def tiled_upscale(
    image: Image.Image,
    upscale_factor: float = 2.0,
    upscale_blend: float = 0.35,
    tile_size: int = UPSCALE_TILE_SIZE,
    overlap: int = UPSCALE_TILE_OVERLAP,
) -> Image.Image:
    """
    Upscale a PIL image using the DRCT-L 4x model with tiled inference.
    Tiles are feather-blended to eliminate seam artifacts.

    Workflow:
        1. Convert PIL → float32 NCHW tensor [0, 1]
        2. Slice into overlapping tile_size×tile_size patches (input-space)
        3. Run each patch through DRCT-L on GPU (output is 4× larger)
        4. Accumulate patches into output buffer using a linear-ramp weight
           kernel (feathering) to blend seams
        5. Divide by accumulated weight → final image
        6. Resize to requested factor with LANCZOS
        7. Optionally blend with plain LANCZOS upscale to reduce hallucinations

    Args:
        image: Input PIL Image (RGB).
        upscale_factor: Target output scale relative to input (0.25–4.0).
                        DRCT-L always runs at 4x; result is resized if needed.
        upscale_blend: Blend amount of AI upscale over plain LANCZOS upscale.
                       0.0 = pure LANCZOS, 1.0 = full AI output.
        tile_size: Tile edge length in input pixels (default 512).
        overlap: Overlap between adjacent tiles in input pixels (default 32).

    Returns:
        Upscaled PIL Image (RGB).
    """
    model = load_upscaler()
    model_scale = model.scale  # 4 for DRCT-L
    upscale_blend = float(max(0.0, min(1.0, upscale_blend)))

    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    # HWC → CHW → NCHW (kept on CPU until each tile is dispatched to GPU)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    in_h, in_w = img_tensor.shape[2], img_tensor.shape[3]
    out_h = in_h * model_scale
    out_w = in_w * model_scale

    # Accumulation buffers live on CPU to keep VRAM free for subsequent tiles
    output = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32)
    weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32)

    step = tile_size - overlap  # stride between tile origins

    def _make_blend_kernel(h: int, w: int, fade: int) -> torch.Tensor:
        """
        Linear-ramp weight mask that feathers the tile edges.
        Values near edges ramp from 0 → 1 over `fade` pixels so that
        overlapping tiles blend smoothly without hard seams.
        """
        kern = torch.ones(h, w, dtype=torch.float32)
        for i in range(fade):
            v = (i + 1) / (fade + 1)
            kern[i, :] *= v
            kern[-(i + 1), :] *= v
            kern[:, i] *= v
            kern[:, -(i + 1)] *= v
        return kern.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Build deduplicated list of tile top-left corners
    def _tile_coords(length: int) -> List[int]:
        coords = list(range(0, max(1, length - tile_size), step))
        last = max(0, length - tile_size)
        if not coords or coords[-1] != last:
            coords.append(last)
        return coords

    y_starts = _tile_coords(in_h)
    x_starts = _tile_coords(in_w)
    total_tiles = len(y_starts) * len(x_starts)

    print(
        f"Tiled upscale: {in_w}x{in_h} → {out_w}x{out_h} "
        f"({total_tiles} tiles, tile={tile_size}px, overlap={overlap}px)"
    )

    for y in y_starts:
        y_end = min(y + tile_size, in_h)
        for x in x_starts:
            x_end = min(x + tile_size, in_w)

            tile = img_tensor[:, :, y:y_end, x:x_end].to(DEVICE)

            with torch.inference_mode():
                tile_out = model(tile).cpu().clamp(0.0, 1.0)

            # Output-space coordinates for this tile
            oy = y * model_scale
            ox = x * model_scale
            oy_end = oy + tile_out.shape[2]
            ox_end = ox + tile_out.shape[3]

            kern = _make_blend_kernel(
                tile_out.shape[2],
                tile_out.shape[3],
                overlap * model_scale,
            )
            output[:, :, oy:oy_end, ox:ox_end] += tile_out * kern
            weight[:, :, oy:oy_end, ox:ox_end] += kern

            del tile, tile_out

    # Normalize by accumulated blend weights
    output = (output / weight.clamp(min=1e-8)).clamp(0.0, 1.0)

    # NCHW float32 → HWC uint8 → PIL
    out_np = (output.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    result_img = Image.fromarray(out_np, "RGB")

    target_w = int(round(in_w * upscale_factor))
    target_h = int(round(in_h * upscale_factor))

    # Resize native 4x output to requested scale (including non-4x factors)
    if (target_w, target_h) != (out_w, out_h):
        result_img = result_img.resize((target_w, target_h), Image.LANCZOS)
        print(f"Resized 4x output → {target_w}x{target_h} ({upscale_factor}x)")

    # Conservative merge: preserve global composition/color from source image.
    # This avoids the "different generation" look while still adding texture.
    if upscale_blend < 1.0:
        base_resized = image.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
        result_img = Image.blend(base_resized, result_img, alpha=upscale_blend)
        print(f"Applied upscale blend: {upscale_blend:.2f}")

    return result_img


def _transfer_high_frequency_details(
    base_image: Image.Image,
    refined_image: Image.Image,
    amount: float,
    blur_radius: float = 1.25,
) -> Image.Image:
    """
    Transfer only micro-detail from refined_image onto base_image.

    This keeps identity, framing, color balance, and overall style from the
    first pass while adding texture from a second generative pass.
    """
    amount = float(max(0.0, min(1.0, amount)))
    if amount <= 0.0:
        return base_image

    base_rgb = base_image.convert("RGB")
    refined_rgb = refined_image.convert("RGB")

    base_np = np.asarray(base_rgb, dtype=np.float32)
    refined_np = np.asarray(refined_rgb, dtype=np.float32)

    # High-frequency band from each image via Gaussian low-pass subtraction.
    base_low = np.asarray(base_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)
    refined_low = np.asarray(refined_rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)
    base_high = base_np - base_low
    refined_high = refined_np - refined_low

    # Inject only incremental detail differences from refined into base.
    mixed = base_np + amount * (refined_high - base_high)
    mixed = np.clip(mixed, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(mixed, "RGB")


def run_2nd_pass(
    image: Image.Image,
    prompt: str,
    strength: float,
    num_steps: int,
    guidance_scale: float,
    shift: float,
    generator: torch.Generator,
    lora_path_loaded: str,
    lora_scale: float,
) -> Image.Image:
    """
    Conservative 2nd-pass detailer using FLUX2 image conditioning.

    Flux2KleinPipeline accepts an `image` conditioning input but does not expose
    a native img2img `strength` parameter. To avoid large style/composition drift:
      1. Run a short, low-guidance conditioned pass.
      2. Transfer only high-frequency detail from that result back to pass-1.

    Args:
        image: Output from the 1st generation pass (PIL RGB).
        prompt: Same prompt used for the 1st pass.
        strength: Detail transfer amount from 2nd-pass result onto pass-1.
        num_steps: Denoising steps for the conditioned 2nd pass.
        guidance_scale: CFG scale for the 2nd pass (keep low; ~1.0–1.3).
        shift: Flow-matching shift value (match 1st pass).
        generator: Seeded RNG — reused for deterministic pass sequencing.
        lora_path_loaded: Loaded LoRA adapter name (empty if none).
        lora_scale: LoRA adapter weight.

    Returns:
        Refined PIL Image.
    """
    global pipeline

    w, h = image.size
    print(
        f"2nd pass detailer: strength={strength}, steps={num_steps}, "
        f"guidance={guidance_scale}, shift={shift}"
    )

    # Match scheduler behavior from pass-1.
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        shift=shift,
    )
    if lora_path_loaded:
        pipeline.set_adapters("flux_lora", adapter_weights=lora_scale)

    t0 = time.time()
    with torch.inference_mode():
        try:
            result = pipeline(
                prompt=prompt,
                image=image,
                height=h,
                width=w,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        except TypeError as e:
            # Backward compatibility for older pipeline builds that may not
            # support image conditioning in __call__.
            if "image" in str(e):
                print("2nd pass skipped: pipeline build does not support image conditioning.")
                return image
            raise

    print(f"2nd pass complete in {time.time() - t0:.2f}s")
    refined = result.images[0]

    # Conservative merge to keep pass-1 identity/style while recovering texture.
    return _transfer_high_frequency_details(image, refined, amount=strength)


def load_lora_weights(
    pipeline: Flux2KleinPipeline,
    lora_path: str,
    lora_scale: float = 1.0
) -> Flux2KleinPipeline:
    """
    Load LoRA weights onto the pipeline.

    Args:
        pipeline: The Flux2Klein pipeline
        lora_path: Path, HuggingFace repo ID, or HTTPS URL for LoRA weights
        lora_scale: Scaling factor for LoRA weights (0.0 to 2.0)

    Returns:
        The pipeline with loaded LoRA weights
    """
    if not lora_path or lora_path.strip() == "":
        return pipeline, False

    print(f"Loading LoRA weights from: {lora_path}")

    try:
        if lora_path.startswith(("http://", "https://")):
            # Download .safetensors from URL to a temporary directory
            print("Downloading LoRA from URL...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, "lora.safetensors")
                urllib.request.urlretrieve(lora_path, tmp_file)
                pipeline.load_lora_weights(
                    tmp_dir, weight_name="lora.safetensors", adapter_name="flux_lora"
                )
        elif os.path.exists(lora_path):
            if lora_path.endswith(".safetensors"):
                lora_dir = os.path.dirname(lora_path) or "."
                lora_name = os.path.basename(lora_path)
                pipeline.load_lora_weights(
                    lora_dir, weight_name=lora_name, adapter_name="flux_lora"
                )
            else:
                pipeline.load_lora_weights(lora_path, adapter_name="flux_lora")
        else:
            # Load from HuggingFace Hub (repo ID)
            pipeline.load_lora_weights(lora_path, adapter_name="flux_lora")

        print(f"LoRA weights loaded successfully")
        return pipeline, True

    except Exception as e:
        print(f"ERROR: Failed to load LoRA weights from '{lora_path}': {e}")
        # Continue without LoRA

    return pipeline, False


def initialize_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    lora_path: str = "",
    lora_scale: float = DEFAULT_LORA_SCALE,
) -> Flux2KleinPipeline:
    """
    Initialize the Flux2Klein pipeline with optional LoRA weights.

    Args:
        model_id: HuggingFace model ID or local path
        lora_path: Optional path to LoRA weights
        lora_scale: LoRA weight scaling factor

    Returns:
        Initialized Flux2KleinPipeline
    """
    global model_loaded

    print(f"Initializing Flux2Klein pipeline with model: {model_id}")

    dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)

    # Always load from the bf16 base pipeline. fp8 is applied via torchao
    # quantization below — this avoids the broken from_single_file conversion
    # path in diffusers (which fails on fp8 scale tensors in the BFL checkpoint
    # format) and is the officially recommended diffusers approach for fp8 FLUX.
    pipeline_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype

    load_kwargs = {"torch_dtype": pipeline_dtype}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    pipeline = Flux2KleinPipeline.from_pretrained(model_id, **load_kwargs)

    # fp8 transformer quantization via optimum-quanto.
    # Reduces transformer VRAM from ~18 GB (bf16) to ~9 GB (fp8), bringing the
    # full pipeline to ~14-16 GB — fits RTX 4090 (24 GB) without any offload.
    # Text encoders remain at bf16 for quality; only the transformer is quantized.
    # Weights are quantized on CPU before pipeline.to(DEVICE) so the GPU move is
    # already at reduced size — no OOM during loading.
    if dtype == torch.float8_e4m3fn:
        # Use qint8 instead of qfloat8. The qfloat8 path in optimum-quanto
        # packs weights into MarlinF8QBytesTensor on CUDA, which requires
        # contiguous inputs — diffusers does not guarantee this, causing
        # "RuntimeError: A is not contiguous". qint8 uses a separate code
        # path with no Marlin dependency and the same ~8-bit storage savings.
        from optimum.quanto import freeze, qint8, quantize
        print("Quantizing transformer to int8 via optimum-quanto (target VRAM ~14-16 GB)")
        quantize(pipeline.transformer, weights=qint8)
        freeze(pipeline.transformer)
        print("int8 quantization complete")

    if ENABLE_CPU_OFFLOAD:
        print("Using model CPU offload")
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(DEVICE)

    # Set scheduler with shift=2.0 — default for photorealistic character output.
    # Lower shift = more time at fine-detail timesteps = natural skin/texture.
    # Overridden per-request in generate_images(); this value only affects warmup.
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        shift=1.5,
    )

    # VAE memory optimizations
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    # Load LoRA if specified — capture success flag so caller can track state
    lora_ok = False
    if lora_path:
        pipeline, lora_ok = load_lora_weights(pipeline, lora_path, lora_scale)

    # Warm up the pipeline
    print("Warming up pipeline...")
    try:
        _ = pipeline(
            prompt="warmup",
            num_inference_steps=1,
            guidance_scale=4.0,
            width=512,
            height=512,
        )
        print("Pipeline warmup complete")
    except Exception as e:
        print(f"Warning: Pipeline warmup failed: {e}")

    model_loaded = True
    return pipeline, lora_ok


# ============================================================================
# Generation Functions
# ============================================================================

def generate_images(job_input: Dict[str, Any], explicit_fields: Optional[set[str]] = None) -> Dict[str, Any]:
    """
    Generate images using FLUX.2 pipeline.

    Args:
        job_input: Validated input parameters
        explicit_fields: Keys explicitly provided by caller before validation.

    Returns:
        Dictionary containing generated images and metadata
    """
    global pipeline, lora_path_loaded
    explicit_fields = explicit_fields or set(job_input.keys())

    # Initialize pipeline if not already loaded
    if pipeline is None:
        model_id = job_input.get("model_id", DEFAULT_MODEL_ID)
        lora_path = job_input.get("lora_path", DEFAULT_LORA_PATH)
        lora_scale = job_input.get("lora_scale", DEFAULT_LORA_SCALE)
        pipeline, lora_ok = initialize_pipeline(model_id, lora_path, lora_scale)
        lora_path_loaded = lora_path if lora_ok else ""

    # Handle LoRA change without reloading the full pipeline
    current_lora_path = job_input.get("lora_path", DEFAULT_LORA_PATH)
    current_lora_scale = job_input.get("lora_scale", DEFAULT_LORA_SCALE)

    if current_lora_path != lora_path_loaded:
        if lora_path_loaded:
            pipeline.unload_lora_weights()
            lora_path_loaded = ""
        if current_lora_path:
            pipeline, lora_ok = load_lora_weights(pipeline, current_lora_path, current_lora_scale)
            if lora_ok:
                lora_path_loaded = current_lora_path

    # Extract parameters - apply preset first, then override with explicit values
    preset_name = job_input.get("preset", "realistic_character")

    # Start with preset values if specified and valid
    if preset_name and preset_name in PRESETS:
        preset = PRESETS[preset_name].copy()
        width = preset.get("width", 1024)
        height = preset.get("height", 1024)
        num_inference_steps = preset.get("num_inference_steps", 35)
        guidance_scale = preset.get("guidance_scale", 2.5)
        shift = preset.get("shift", 2.0)
    else:
        width = 1024
        height = 1024
        num_inference_steps = 35
        guidance_scale = 2.0
        shift = 1.5

    # Override with explicit values if provided
    prompt = job_input["prompt"]
    if "width" in explicit_fields:
        width = job_input["width"]
    if "height" in explicit_fields:
        height = job_input["height"]
    if "num_inference_steps" in explicit_fields:
        num_inference_steps = job_input["num_inference_steps"]
    if "guidance_scale" in explicit_fields:
        guidance_scale = job_input["guidance_scale"]
    if "shift" in explicit_fields:
        shift = job_input["shift"]

    seed = job_input.get("seed", -1)
    num_images = job_input.get("num_images", 1)
    output_format = job_input.get("output_format", "png")

    # Prompt enhancement parameters
    prompt_2 = job_input.get("prompt_2", "")
    prompt_2_weight = job_input.get("prompt_2_weight", 0.0)

    # Set random seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        seed = int(time.time() * 1000) % (2**31)
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # shift controls noise schedule bias: lower = more organic detail/skin texture,
    # higher = stronger large-structure coherence. 2.0 is the photorealism default.
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        shift=shift,
    )

    # Apply LoRA scale before each inference call.
    # Only call set_adapters when the adapter is actually registered (lora_path_loaded
    # is only set on successful load, so this guards against silent load failures).
    if lora_path_loaded:
        pipeline.set_adapters("flux_lora", adapter_weights=current_lora_scale)

    # -------------------------------------------------------------------------
    # Prompt Enhancement: Weighted blending of two prompts
    # -------------------------------------------------------------------------
    # FLUX uses T5 encoder. We create weighted embeddings by:
    # 1. Encode both prompts separately
    # 2. Blend: final = prompt_embeds * (1 - weight) + prompt_2_embeds * weight
    prompt_embeds = None
    pooled_prompt_embeds = None

    if prompt_2 and prompt_2_weight > 0:
        print(f"Prompt weighting: blending prompt with prompt_2 at weight {prompt_2_weight}")

        # Encode primary prompt
        prompt_1_embeds, pooled_1_embeds, _, _ = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device="cuda",
            num_images_per_prompt=num_images,
        )

        # Encode secondary prompt
        prompt_2_embeds, pooled_2_embeds, _, _ = pipeline.encode_prompt(
            prompt=prompt_2,
            prompt_2=None,
            device="cuda",
            num_images_per_prompt=num_images,
        )

        # Weighted blend
        w = prompt_2_weight
        prompt_embeds = prompt_1_embeds * (1 - w) + prompt_2_embeds * w
        pooled_prompt_embeds = pooled_1_embeds * (1 - w) + pooled_2_embeds * w

        print(f"Prompt weighting applied: {prompt} + {prompt_2} (weight={w})")

    print(f"Generating image(s): {width}x{height}, steps={num_inference_steps}, "
          f"guidance={guidance_scale}, shift={shift}")

    # Generate images
    start_time = time.time()

    with torch.inference_mode():
        # Use prompt_embeds if weighted blending was applied, otherwise use text prompt
        if prompt_embeds is not None:
            result = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
            )
        else:
            result = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images,
            )

    generation_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # Post-processing: 2nd pass detailer and/or tiled upscale
    # -------------------------------------------------------------------------
    enable_2nd_pass = job_input.get("enable_2nd_pass", False)
    second_pass_strength = job_input.get("second_pass_strength", 0.2)
    second_pass_steps = job_input.get("second_pass_steps", 12)
    second_pass_guidance_scale = job_input.get("second_pass_guidance_scale", 1.0)
    enable_upscale = job_input.get("enable_upscale", False)
    upscale_factor = job_input.get("upscale_factor", 2.0)
    upscale_blend = job_input.get("upscale_blend", 0.35)

    processed_images = []
    for img in result.images:
        if enable_2nd_pass:
            img = run_2nd_pass(
                image=img,
                prompt=prompt,
                strength=second_pass_strength,
                num_steps=second_pass_steps,
                guidance_scale=second_pass_guidance_scale,
                shift=shift,
                generator=generator,
                lora_path_loaded=lora_path_loaded,
                lora_scale=current_lora_scale,
            )
        if enable_upscale:
            img = tiled_upscale(
                img,
                upscale_factor=upscale_factor,
                upscale_blend=upscale_blend,
            )
        processed_images.append(img)

    # Process output images
    return_type = job_input.get("return_type", "s3" if S3_BUCKET_NAME else "base64")

    if return_type == "s3":
        # Upload to S3 and return presigned URLs
        image_urls = []
        for img in processed_images:
            url = upload_to_s3(img, output_format)
            if url:
                image_urls.append(url)
            else:
                # Fallback to base64 if S3 upload fails
                image_urls.append(encode_image_to_base64(img, output_format))
        
        response = {
            "image_urls": image_urls,
            "format": output_format,
            "return_type": "s3",
            "parameters": {
                "prompt": prompt,
                "prompt_2": prompt_2 if prompt_2 else None,
                "prompt_2_weight": prompt_2_weight if prompt_2 else None,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "shift": shift,
                "enable_2nd_pass": enable_2nd_pass,
                "second_pass_strength": second_pass_strength if enable_2nd_pass else None,
                "second_pass_steps": second_pass_steps if enable_2nd_pass else None,
                "second_pass_guidance_scale": second_pass_guidance_scale if enable_2nd_pass else None,
                "enable_upscale": enable_upscale,
                "upscale_factor": upscale_factor if enable_upscale else None,
                "upscale_blend": upscale_blend if enable_upscale else None,
            },
            "metadata": {
                "model_id": job_input.get("model_id", DEFAULT_MODEL_ID),
                "lora_path": current_lora_path if current_lora_path else None,
                "lora_scale": current_lora_scale if current_lora_path else None,
                "generation_time": f"{generation_time:.2f}s",
                "preset": preset_name if preset_name else None,
                "s3_bucket": S3_BUCKET_NAME,
                "presigned_url_expiry_seconds": S3_PRESIGNED_URL_EXPIRY,
            }
        }
    else:
        # Return base64 encoded images
        images_base64 = []
        for img in processed_images:
            images_base64.append(encode_image_to_base64(img, output_format))

        response = {
            "images": images_base64,
            "format": output_format,
            "return_type": "base64",
            "parameters": {
                "prompt": prompt,
                "prompt_2": prompt_2 if prompt_2 else None,
                "prompt_2_weight": prompt_2_weight if prompt_2 else None,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "shift": shift,
                "enable_2nd_pass": enable_2nd_pass,
                "second_pass_strength": second_pass_strength if enable_2nd_pass else None,
                "second_pass_steps": second_pass_steps if enable_2nd_pass else None,
                "second_pass_guidance_scale": second_pass_guidance_scale if enable_2nd_pass else None,
                "enable_upscale": enable_upscale,
                "upscale_factor": upscale_factor if enable_upscale else None,
                "upscale_blend": upscale_blend if enable_upscale else None,
            },
            "metadata": {
                "model_id": job_input.get("model_id", DEFAULT_MODEL_ID),
                "lora_path": current_lora_path if current_lora_path else None,
                "lora_scale": current_lora_scale if current_lora_path else None,
                "generation_time": f"{generation_time:.2f}s",
                "preset": preset_name if preset_name else None,
            }
        }

    return response


# ============================================================================
# Main Handler
# ============================================================================

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod serverless handler function.

    Args:
        job: RunPod job dictionary with 'input' key containing parameters

    Returns:
        Dictionary with generated images or error message
    """
    try:
        job_input = job.get("input", {})
        explicit_fields = set(job_input.keys())

        # Validate input
        validation_result = validate(job_input, INPUT_SCHEMA)
        if "errors" in validation_result:
            return {
                "error": "Validation failed",
                "details": validation_result["errors"]
            }

        validated_input = validation_result["validated_input"]

        # Generate images
        result = generate_images(validated_input, explicit_fields=explicit_fields)

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }


# ============================================================================
# Serverless Entry Point
# ============================================================================

if __name__ == "__main__":
    print("Starting FLUX.2-klein RunPod Serverless Worker...")
    print(f"Model ID: {DEFAULT_MODEL_ID}")
    print(f"Default LoRA Path: {DEFAULT_LORA_PATH if DEFAULT_LORA_PATH else 'None'}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Flash Attention: {USE_FLASH_ATTN} (note: not supported by Flux2KleinPipeline attn kwarg)")
    print(f"S3 Bucket: {S3_BUCKET_NAME if S3_BUCKET_NAME else 'Not configured'}")
    print(f"HF Token: {'Configured' if HF_TOKEN else 'Not configured (required for gated models)'}")

    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
