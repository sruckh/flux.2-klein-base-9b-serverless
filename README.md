# FLUX.2-klein-base-9B RunPod Serverless

[![RunPod](https://img.shields.io/badge/RunPod-Serverless-blue)](https://runpod.io)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A production-ready RunPod serverless deployment for FLUX.2-klein-base-9B image generation with custom LoRA weight support, tuned for photorealistic human portraits.

This serverless API generates high-quality images using the **FLUX.2-klein-base-9B** base model via `Flux2KleinPipeline`, with support for custom LoRA weights loaded from HTTPS URLs, HuggingFace Hub, or local paths. Output is delivered via S3 presigned URLs or base64.

> **Important:** This uses the **base (undistilled)** model. Use 35–50 inference steps and guidance scale 2.0–3.0. Do **not** use 4–8 steps or guidance 1.0 — those settings are for the distilled variant. Do **not** use guidance 4.0+ — it produces over-rendered, plastic-looking skin.

## ✨ Features

- **FLUX.2-klein-base-9B** - Undistilled 9B-parameter flow-match transformer by Black Forest Labs
- **Flux2KleinPipeline** - Correct pipeline class via git diffusers (not available in stable releases)
- **Custom LoRA Support** - Load LoRA weights from HTTPS URLs, HuggingFace Hub, or local `.safetensors`
- **LoRA Hot-Swap** - Switch LoRA between requests without reloading the base model
- **Tunable Scheduler Shift** - `FlowMatchEulerDiscreteScheduler` with configurable shift (default 1.5)
- **Photorealism-Tuned Presets** - Low guidance + low shift defaults for natural skin texture
- **INT8 Quantization** - Transformer quantized via optimum-quanto for ~9 GB VRAM footprint
- **2nd Pass Detailer** - Optional low-denoise img2img pass (same pipeline, zero extra VRAM) for skin pore and micro-contrast refinement — equivalent to a ComfyUI KSampler detailer pass
- **Tiled 4× Upscaler** - Optional DRCT-L super-resolution via [4xRealWebPhoto_v4_drct-l](https://github.com/Phhofm/models) with feather-blended 512px tiles; resizes to target factor (e.g. 2× output from a 4× SR pass)
- **Flexible Generation** - Configurable resolution, steps, guidance scale, shift, and batch size
- **Multiple Output Formats** - PNG, JPEG, WebP support
- **S3 Storage** - Upload images to S3 with presigned URLs (no base64 size limits)
- **RunPod Model Cache** - HuggingFace cache on network volumes for fast restarts; upscaler model auto-downloaded and cached on first use
- **High-Performance Downloads** - Xet storage backend with concurrent chunk downloads

## 🏗️ Architecture

The system uses `Flux2KleinPipeline` (git diffusers) with `FlowMatchEulerDiscreteScheduler`. The pipeline is loaded once per worker into full VRAM and reused across requests. The transformer is quantized to INT8 via optimum-quanto on CPU before GPU placement, reducing VRAM from ~18 GB to ~9 GB. LoRA weights are hot-swapped per request using `unload_lora_weights()` + reload, avoiding full model reinitialization.

### Data Flow Pipeline

1. **Input Validation** — Parameters validated against schema, preset defaults applied
2. **Pipeline Init** — `Flux2KleinPipeline.from_pretrained()` + INT8 quantization + `.to("cuda")`; scheduler initialized with `shift=1.5`; VAE slicing/tiling enabled
3. **LoRA Hot-Swap** — If `lora_path` changed since last request, unload previous and load new
4. **Scheduler Update** — `FlowMatchEulerDiscreteScheduler.from_config(config, shift=N)` applied per request
5. **Image Generation** — Flow-match inference via `Flux2KleinPipeline`
6. **2nd Pass Detailer** *(optional)* — Low-denoise img2img pass on the same pipeline with `strength=0.3`; adds fine skin texture and micro-contrast without drifting from the original composition; LoRA stays active for subject consistency
7. **Tiled Upscale** *(optional)* — DRCT-L 4× SR via spandrel with 512px/32px feather-blended tiles; result resized to `upscale_factor` via LANCZOS; model auto-downloaded to `/runpod-volume/models/` on first use
8. **Output** — Images uploaded to S3 (presigned URL) or encoded to base64

## 🚀 Quick Start

### Prerequisites

- **Docker** - For containerized deployment
- **RunPod Account** - For serverless GPU deployment
- **NVIDIA GPU** - H100 or A100 80GB recommended; RTX 4090 (24 GB) works with INT8 quantization
- **HuggingFace Token** - Required (FLUX.2-klein-base-9B is a gated model)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/flux.2-klein-serverless.git
cd flux.2-klein-serverless

# Build the Docker image
docker build -t flux-2-klein-serverless .

# Run locally (GPU required)
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  -e S3_BUCKET_NAME=your-bucket \
  flux-2-klein-serverless

# Test the endpoint
curl -X POST http://localhost:8000/runpod/v1/lgpu/input \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Deploy to RunPod (GitHub Integration)

The easiest deployment method uses RunPod's GitHub integration for automatic builds on every push.

1. **Push code to GitHub:**
```bash
git init
git add .
git commit -m "Initial FLUX.2-klein serverless implementation"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. **Create Serverless Template** at [console.runpod.io/serverless/user/templates](https://console.runpod.io/serverless/user/templates):
   - Select **GitHub** as source
   - Choose your repository
   - Configure: **Container Disk**: 30 GB, **Min/Max Workers**: 0-10, **GPU**: NVIDIA H100
   - **Network Volume**: Recommended for model caching (30GB+)

3. **Set Environment Variables** (see Configuration section below)

4. **Deploy** — RunPod automatically builds and deploys. Future pushes trigger redeployment.

## 📖 Documentation

### Optimized Presets

All presets are tuned for photorealistic human character output. Lower guidance and shift produce natural skin texture; higher values produce more stylised or structured results.

| Preset | Steps | Guidance | Shift | Resolution | Best For |
|--------|-------|----------|-------|------------|----------|
| `realistic_character` | 35 | 2.0 | 1.5 | 1024×1024 | **Photorealistic human LoRAs** (default) |
| `portrait_hd` | 40 | 2.0 | 1.5 | 1024×1536 | High-detail vertical portraits |
| `cinematic_full` | 35 | 2.5 | 1.5 | 1536×1024 | Full-body cinematic compositions |
| `fast_preview` | 20 | 2.0 | 1.5 | 1024×1024 | Quick prompt/seed testing |
| `maximum_quality` | 50 | 2.5 | 1.0 | 1024×1024 | Highest quality final output |

### Realistic Character Best Practices

> **This is the base (undistilled) model.** Do not use distilled model settings.

**Optimal Inference Settings:**
- **Steps:** 35–50 (sweet spot: 35–40 for quality/speed balance)
- **Guidance:** 2.0–2.5 — lower = more organic, natural-looking skin; higher = more stylised
- **Scheduler Shift:** 1.0–1.5 for photorealism; lower = more denoising budget on fine-detail timesteps (pores, texture, natural imperfection)
- **Resolution:** 1024×1024 standard, 1024×1536 for portraits (match your training preview size)
- **LoRA Scale:** 0.7–0.9 — start at 0.85; avoid 1.0+ (pushes identity too hard, reduces naturalness)

**Trigger Word:** Include your LoRA trigger word first in the prompt (AI-Toolkit convention).

**Prompt Structure:**
```
[TRIGGER] [subject description], [clothing/environment], [lighting], [camera/lens], [film language]
```

Example:
```
TOK woman, close-up portrait, soft window light from the left, slight smile,
out-of-focus office interior background, shot on Sony A7IV, 105mm f/2.8,
natural skin, visible pores, unretouched, ISO 800, slight film grain
```

**Prompting Notes:**
- FLUX.2-klein uses Qwen3-8B as text encoder — it handles long, descriptive prompts well
- Avoid SDXL boilerplate ("masterpiece, best quality", "photorealistic, highly detailed skin texture") — these keywords are associated with over-processed, retouched aesthetics
- Use camera/film language instead: "ISO 800", "slight film grain", "unretouched", "candid" — these anchor the model to real photography priors
- FLUX.2-klein does **not** support `negative_prompt` — it is ignored

### Scheduler Shift Reference

| Shift | Effect |
|-------|--------|
| `1.0` | Maximum fine-detail budget; most natural skin/texture; best for close-up portraits |
| `1.5` | **Default for photorealistic character LoRAs**; natural texture with good structure |
| `2.0` | Balanced; slightly stronger large-structure coherence |
| `3.0` | Better face/structure coherence; may smooth skin texture |
| `5.0` | Strong structure emphasis; helps complex lighting or multi-figure scenes |

### API Reference

**Request Format:**
```json
{
  "input": {
    "prompt": "TOK woman, close-up portrait, soft window light, Sony A7IV, 105mm f/2.8, natural skin, visible pores, unretouched, ISO 800",
    "preset": "portrait_hd",
    "lora_path": "https://example.com/your-character-lora.safetensors",
    "lora_scale": 0.85,
    "guidance_scale": 2.0,
    "shift": 1.5,
    "seed": 42,
    "return_type": "s3",
    "enable_2nd_pass": true,
    "second_pass_strength": 0.3,
    "second_pass_steps": 20,
    "enable_upscale": true,
    "upscale_factor": 2.0
  }
}
```

**Response Format (S3 - default if configured):**
```json
{
  "image_urls": ["https://s3.amazonaws.com/bucket/flux2-klein/uuid.jpg?X-Amz-Algorithm=..."],
  "format": "jpeg",
  "return_type": "s3",
  "parameters": { "width": 1024, "height": 1536, "num_inference_steps": 40, "guidance_scale": 2.0, "shift": 1.5, "seed": 42 },
  "metadata": {
    "model_id": "black-forest-labs/FLUX.2-klein-base-9B",
    "generation_time": "12.34s",
    "preset": "portrait_hd",
    "s3_bucket": "your-bucket-name",
    "presigned_url_expiry_seconds": 3600
  }
}
```

**Response Format (Base64 - fallback):**
```json
{
  "images": ["<base64_encoded_image>"],
  "format": "jpeg",
  "return_type": "base64",
  "parameters": { "width": 1024, "height": 1536, "num_inference_steps": 40, "guidance_scale": 2.0, "shift": 1.5, "seed": 42 },
  "metadata": {
    "model_id": "black-forest-labs/FLUX.2-klein-base-9B",
    "generation_time": "12.34s",
    "preset": "portrait_hd"
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text prompt — include trigger word first |
| `preset` | string | `realistic_character` | Quality preset (see table above) |
| `width` | int | `1024` | Image width (multiple of 16) |
| `height` | int | `1024` | Image height (multiple of 16) |
| `num_inference_steps` | int | `35` | Denoising steps — use 35–50 for base model |
| `guidance_scale` | float | `2.0` | CFG scale — 2.0–2.5 for photorealism; higher = more stylised |
| `shift` | float | `1.5` | Scheduler shift — 1.0–1.5 for natural skin texture; higher for structure |
| `seed` | int | `-1` | Random seed (-1 for random) |
| `num_images` | int | `1` | Number of images per request (1–4) |
| `output_format` | string | `"jpeg"` | Output format: `png`, `jpeg`, `webp` |
| `return_type` | string | `"s3"` | Response type: `s3` (presigned URL) or `base64` |
| `lora_path` | string | `""` | HuggingFace repo ID, local path, or HTTPS URL to `.safetensors` |
| `lora_scale` | float | `1.0` | LoRA weight scale (0.0–2.0); recommended 0.75–0.9 |
| `enable_2nd_pass` | bool | `false` | Enable low-denoise img2img detailer pass after generation |
| `second_pass_strength` | float | `0.3` | Detailer denoise strength (0.05–0.95); 0.3 adds skin/pore detail without composition drift |
| `second_pass_steps` | int | `20` | Inference steps for the 2nd pass (5–50); actual steps applied = `steps × strength` |
| `enable_upscale` | bool | `false` | Enable DRCT-L 4× tiled super-resolution upscaling |
| `upscale_factor` | float | `2.0` | Target upscale multiplier (0.25–4.0); model runs at 4×, result resized to this factor |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_ID` | `black-forest-labs/FLUX.2-klein-base-9B` | Base model (loaded via `Flux2KleinPipeline`) |
| `DEFAULT_LORA_PATH` | `""` | Default LoRA loaded at worker startup |
| `DEFAULT_LORA_SCALE` | `1.0` | Default LoRA weight scale |
| `DEVICE` | `cuda` | Compute device |
| `DTYPE` | `float8_e4m3fn` | Compute dtype — triggers INT8 quantization of transformer via optimum-quanto |
| `ENABLE_CPU_OFFLOAD` | `false` | Use CPU offload instead of full VRAM placement |
| `HF_TOKEN` | `""` | **Required** — FLUX.2-klein-base-9B is a gated model |
| `UPSCALER_MODEL_PATH` | `/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth` | Path to DRCT-L upscaler model; auto-downloaded from GitHub on first use |

### S3 Configuration (Optional)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `S3_BUCKET_NAME` | `""` | S3 bucket for image storage |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_ACCESS_KEY_ID` | `""` | AWS access key |
| `S3_SECRET_ACCESS_KEY` | `""` | AWS secret key |
| `S3_ENDPOINT_URL` | `""` | Custom S3 endpoint (e.g., MinIO, Wasabi) |
| `S3_PRESIGNED_URL_EXPIRY` | `3600` | Presigned URL expiry in seconds |

**Note:** When `S3_BUCKET_NAME` is configured, images are uploaded to S3 and a presigned URL is returned instead of base64. This avoids payload size limits and improves response times.

### HuggingFace Performance Tuning

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HF_HOME` | `/runpod-volume/huggingface` | HuggingFace cache directory |
| `HF_HUB_CACHE` | `/runpod-volume/huggingface/hub` | Model cache directory |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Enable high-performance Xet downloads |
| `HF_XET_NUM_CONCURRENT_RANGE_GETS` | `32` | Concurrent download chunks |
| `HF_HUB_ETAG_TIMEOUT` | `30` | Timeout for metadata requests (seconds) |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `300` | Timeout for file downloads (seconds) |

### VRAM Requirements

| Configuration | VRAM | Notes |
|---------------|------|-------|
| BF16, no offload | ~22–24 GB | Best quality; H100/A100 80GB |
| INT8 quantized (default) | ~14–16 GB | Good quality; RTX 4090 (24 GB), A100 40 GB |
| INT8 + 2nd pass | ~14–16 GB | Same pipeline reused — zero additional VRAM |
| INT8 + upscaler | ~14–16 GB + ~300 MB | DRCT-L loaded alongside; tiled inference keeps per-tile VRAM trivial |
| BF16 + CPU offload | ~12–14 GB | Slower; set `ENABLE_CPU_OFFLOAD=true` |

The default `DTYPE=float8_e4m3fn` triggers INT8 quantization of the transformer via optimum-quanto, reducing transformer VRAM from ~18 GB to ~9 GB while keeping text encoders and VAE at BF16.

The 2nd pass detailer reuses the already-loaded `Flux2KleinPipeline` with an `image` + `strength` parameter — no second model is loaded, so VRAM usage is identical to a standard generation request. The DRCT-L upscaler (~300 MB) is loaded separately via spandrel and cached globally after first use; tiled inference keeps peak per-tile GPU memory negligible.

## 🎨 Training Custom LoRAs

Train LoRA weights using [ai-toolkit](https://github.com/ostris/ai-toolkit):

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt
python flux_train_ui.py
```

**Recommended Training Config for Realistic Characters:**
- **Network:** linear=128, linear_alpha=64, conv=64, conv_alpha=32
- **Steps:** 7000 (BFL recommendation for photorealistic)
- **Learning Rate:** 0.000095
- **Weight Decay:** 0.00001
- **Dataset:** 20–40 images with diverse angles, expressions, and lighting

**Training → Inference Alignment:** Match `num_inference_steps` and `lora_scale` to your AI-Toolkit training YAML's `sample.sample_steps` and `sample.lora_weight`. For `guidance_scale`, start at 2.0 regardless of training YAML value — the training guidance is for the distilled sampler and does not translate directly.

## 📄 License

This code is provided as-is for RunPod serverless deployments. The FLUX.2-klein-base-9B model has its own license terms — refer to the [HuggingFace model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B).

## 🙏 Acknowledgments

- **[ai-toolkit](https://github.com/ostris/ai-toolkit)** by ostris - Training framework and reference implementation
- **[FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)** by Black Forest Labs - Base model
- **[RunPod](https://runpod.io)** - Serverless GPU infrastructure

## 📧 Support

- **This deployment:** Check logs in RunPod console
- **Model behavior:** Consult [FLUX.2-klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- **Training:** See [ai-toolkit issues](https://github.com/ostris/ai-toolkit/issues)
