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
import time
import uuid
from typing import Dict, Any, List, Optional, Union

import boto3
import torch
from botocore.exceptions import ClientError
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux_output import FluxPipelineOutput
from PIL import Image
from safetensors.torch import load_file

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
    "black-forest-labs/FLUX.2-klein-9B"
)

DEFAULT_LORA_PATH = os.getenv("DEFAULT_LORA_PATH", "")
DEFAULT_LORA_SCALE = float(os.getenv("DEFAULT_LORA_SCALE", "1.0"))

DEVICE = os.getenv("DEVICE", "cuda")
DTYPE = os.getenv("DTYPE", "bf16").lower()
USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN", "true").lower() == "true"

# Map dtype string to torch dtype
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# ============================================================================
# Optimized Presets for Different Use Cases
# Based on research from:
# - Black Forest Labs FLUX.2 documentation
# - fal.ai FLUX.2 training guides
# - Community testing and best practices
# ============================================================================

PRESETS = {
    # Ultra-realistic human character/portrait LoRA
    # Optimized for trained character LoRAs to achieve maximum photorealism
    # Based on BFL recommendation: 20-30 steps for photorealistic styles
    "realistic_character": {
        "num_inference_steps": 28,
        "guidance_scale": 2.5,
        "width": 1024,
        "height": 1024,
        "max_sequence_length": 512,
        "description": "Ultra-realistic human character with natural lighting"
    },

    # High-quality portrait with enhanced detail
    # More steps for finer skin texture and detail
    "portrait_hd": {
        "num_inference_steps": 30,
        "guidance_scale": 3.0,
        "width": 1024,
        "height": 1536,  # 2:3 portrait ratio
        "max_sequence_length": 512,
        "description": "High-detail portrait with vertical composition"
    },

    # Cinematic full-body character shot
    # Wider aspect ratio for full-body compositions
    "cinematic_full": {
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "width": 1536,
        "height": 1024,  # 3:2 cinematic ratio
        "max_sequence_length": 512,
        "description": "Cinematic full-body character composition"
    },

    # Fast iteration for testing
    # Lower steps for quick generation during prompt development
    "fast_preview": {
        "num_inference_steps": 15,
        "guidance_scale": 3.0,
        "width": 1024,
        "height": 1024,
        "max_sequence_length": 512,
        "description": "Fast preview for prompt testing"
    },

    # Maximum quality for final output
    # Higher steps for best possible image quality
    "maximum_quality": {
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
        "width": 1024,
        "height": 1024,
        "max_sequence_length": 512,
        "description": "Maximum quality generation (slower)"
    },
}


# ============================================================================
# Global Pipeline Instance
# ============================================================================

pipeline: Optional[FluxPipeline] = None
model_loaded = False


# ============================================================================
# Input Validation Schema
# ============================================================================

INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
        "constraints": lambda x: len(x.strip()) > 0,
    },
    "negative_prompt": {
        "type": str,
        "required": False,
        "default": "",
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
        "default": 2.5,
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


def load_lora_weights(
    pipeline: FluxPipeline,
    lora_path: str,
    lora_scale: float = 1.0
) -> FluxPipeline:
    """
    Load LoRA weights onto the pipeline.

    Args:
        pipeline: The FLUX pipeline
        lora_path: Path or HuggingFace repo ID for LoRA weights
        lora_scale: Scaling factor for LoRA weights (0.0 to 2.0)

    Returns:
        The pipeline with loaded LoRA weights
    """
    if not lora_path or lora_path.strip() == "":
        return pipeline

    print(f"Loading LoRA weights from: {lora_path}")

    try:
        # Check if it's a local file path
        if os.path.exists(lora_path):
            if lora_path.endswith(".safetensors"):
                # Load safetensors file directly
                state_dict = load_file(lora_path)
                # Apply weights to the transformer
                pipeline.transformer.load_state_dict(state_dict, strict=False)
            else:
                # Try loading as HuggingFace repo style
                pipeline.load_lora_weights(lora_path)
        else:
            # Load from HuggingFace Hub
            pipeline.load_lora_weights(lora_path)

        # Set adapter scale
        if hasattr(pipeline, "set_adapters"):
            pipeline.set_adapters(["default"], adapter_weights=[lora_scale])
        elif hasattr(pipeline, "_lora_scale"):
            pipeline._lora_scale = lora_scale

        print(f"LoRA weights loaded with scale: {lora_scale}")

    except Exception as e:
        print(f"Warning: Failed to load LoRA weights: {e}")
        # Continue without LoRA

    return pipeline


def initialize_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    lora_path: str = "",
    lora_scale: float = DEFAULT_LORA_SCALE,
) -> FluxPipeline:
    """
    Initialize the FLUX.2 pipeline with optional LoRA weights.

    Args:
        model_id: HuggingFace model ID or local path
        lora_path: Optional path to LoRA weights
        lora_scale: LoRA weight scaling factor

    Returns:
        Initialized FluxPipeline
    """
    global model_loaded

    print(f"Initializing FLUX pipeline with model: {model_id}")

    # Get torch dtype
    torch_dtype = DTYPE_MAP.get(DTYPE, torch.bfloat16)

    # Load the pipeline with HF token for gated models
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "use_flash_attention_2": USE_FLASH_ATTN,
        "device_map": "auto",
    }
    
    # Add token if provided (for gated models like black-forest-labs/FLUX.2-klein-9B)
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN
    
    pipeline = FluxPipeline.from_pretrained(model_id, **load_kwargs)

    # Load LoRA if specified
    if lora_path:
        pipeline = load_lora_weights(pipeline, lora_path, lora_scale)

    # Enable memory optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    # Warm up the pipeline
    print("Warming up pipeline...")
    try:
        _ = pipeline(
            prompt="warmup",
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512,
            max_sequence_length=256,
        )
        print("Pipeline warmup complete")
    except Exception as e:
        print(f"Warning: Pipeline warmup failed: {e}")

    model_loaded = True
    return pipeline


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """
    Calculate dynamic shift for FLUX flow matching scheduler.

    This matches the dynamic shifting used in FLUX models during training
    as implemented in ai-toolkit.

    Args:
        image_seq_len: Length of the image sequence (h * w / patch_size^2)
        base_seq_len: Base sequence length (default 256)
        max_seq_len: Maximum sequence length (default 4096)
        base_shift: Base shift value (default 0.5)
        max_shift: Maximum shift value (default 1.16)

    Returns:
        Calculated shift value
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# ============================================================================
# Generation Functions
# ============================================================================

def generate_images(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate images using FLUX.2 pipeline.

    Args:
        job_input: Validated input parameters

    Returns:
        Dictionary containing generated images and metadata
    """
    global pipeline

    # Initialize pipeline if not already loaded
    if pipeline is None:
        model_id = job_input.get("model_id", DEFAULT_MODEL_ID)
        lora_path = job_input.get("lora_path", "")
        lora_scale = job_input.get("lora_scale", DEFAULT_LORA_SCALE)
        pipeline = initialize_pipeline(model_id, lora_path, lora_scale)

    # Handle model or LoRA change
    current_lora_path = job_input.get("lora_path", "")
    current_lora_scale = job_input.get("lora_scale", DEFAULT_LORA_SCALE)

    # Reinitialize if LoRA path changed
    if current_lora_path != DEFAULT_LORA_PATH:
        pipeline = initialize_pipeline(
            job_input.get("model_id", DEFAULT_MODEL_ID),
            current_lora_path,
            current_lora_scale
        )

    # Extract parameters - apply preset first, then override with explicit values
    preset_name = job_input.get("preset", "realistic_character")

    # Start with preset values if specified and valid
    if preset_name and preset_name in PRESETS:
        preset = PRESETS[preset_name].copy()
        width = preset.get("width", 1024)
        height = preset.get("height", 1024)
        num_inference_steps = preset.get("num_inference_steps", 28)
        guidance_scale = preset.get("guidance_scale", 2.5)
        max_sequence_length = preset.get("max_sequence_length", 512)
    else:
        width = 1024
        height = 1024
        num_inference_steps = 28
        guidance_scale = 2.5
        max_sequence_length = 512

    # Override with explicit values if provided
    prompt = job_input["prompt"]
    negative_prompt = job_input.get("negative_prompt", "")
    if "width" in job_input:
        width = job_input["width"]
    if "height" in job_input:
        height = job_input["height"]
    if "num_inference_steps" in job_input:
        num_inference_steps = job_input["num_inference_steps"]
    if "guidance_scale" in job_input:
        guidance_scale = job_input["guidance_scale"]
    if "max_sequence_length" in job_input:
        max_sequence_length = job_input["max_sequence_length"]

    seed = job_input.get("seed", -1)
    num_images = job_input.get("num_images", 1)
    output_format = job_input.get("output_format", "png")

    # Set random seed
    generator = None
    if seed >= 0:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        seed = int(time.time() * 1000) % (2**31)
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Calculate dynamic shift based on image size
    # FLUX uses patch_size = 2, so effective seq_len = (h/2) * (w/2)
    h_patches = height // 2
    w_patches = width // 2
    image_seq_len = h_patches * w_patches

    shift = calculate_shift(image_seq_len)

    print(f"Generating image(s): {width}x{height}, steps={num_inference_steps}, "
          f"guidance={guidance_scale}, shift={shift:.4f}")

    # Generate images
    start_time = time.time()

    with torch.inference_mode():
        result: FluxPipelineOutput = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
            max_sequence_length=max_sequence_length,
            # FLUX-specific parameters
            joint_attention_kwargs={"scale": 1.0},
        )

    generation_time = time.time() - start_time

    # Process output images
    return_type = job_input.get("return_type", "s3" if S3_BUCKET_NAME else "base64")
    
    if return_type == "s3":
        # Upload to S3 and return presigned URLs
        image_urls = []
        for img in result.images:
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
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "max_sequence_length": max_sequence_length,
                "shift": shift,
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
        for img in result.images:
            images_base64.append(encode_image_to_base64(img, output_format))
        
        response = {
            "images": images_base64,
            "format": output_format,
            "return_type": "base64",
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "num_images": num_images,
                "max_sequence_length": max_sequence_length,
                "shift": shift,
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

        # Validate input
        validation_result = validate(job_input, INPUT_SCHEMA)
        if "errors" in validation_result:
            return {
                "error": "Validation failed",
                "details": validation_result["errors"]
            }

        validated_input = validation_result["validated_input"]

        # Generate images
        result = generate_images(validated_input)

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
    print(f"Flash Attention: {USE_FLASH_ATTN}")
    print(f"S3 Bucket: {S3_BUCKET_NAME if S3_BUCKET_NAME else 'Not configured'}")
    print(f"HF Token: {'Configured' if HF_TOKEN else 'Not configured (required for gated models)'}")

    # Start the serverless worker
    runpod.serverless.start({"handler": handler})
