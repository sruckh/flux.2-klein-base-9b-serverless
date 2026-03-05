# FLUX.2-klein-9B RunPod Serverless

[![RunPod](https://img.shields.io/badge/RunPod-Serverless-blue)](https://runpod.io)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> A production-ready RunPod serverless deployment for **FLUX.2-klein-9B** (distilled) image generation with custom LoRA weight support, tuned for photorealistic human portraits.

This serverless API generates high-quality images using the **FLUX.2-klein-9B distilled** model via `Flux2KleinPipeline`, loaded as separate FP8 components (transformer + abliterated Qwen3-8B text encoder + VAE) to minimize VRAM. Output is delivered via S3 presigned URLs or base64.

> **Model:** This uses the **distilled** (step + guidance distilled) variant. Optimal settings are **4–16 steps, guidance 1.0**. The distilled model produces excellent results in as few as 4 steps; 8–12 steps gives a good quality/speed balance.

## ✨ Features

- **FLUX.2-klein-9B Distilled** — Step + guidance distilled 9B-parameter flow-match transformer by Black Forest Labs
- **FP8 Component Loading** — Transformer (FP8) + Qwen3-8B text encoder (FP8, abliterated) + VAE loaded separately; fits in ~24 GB VRAM with `enable_model_cpu_offload()`
- **Abliterated Text Encoder** — `edicamargo/qwen_3_8b_fp8mixed_abliterated` weights with `Qwen/Qwen3-8B-FP8` config; removes refusal restrictions for creative generation
- **Flux2KleinPipeline** — Correct pipeline class via git diffusers (not available in stable releases)
- **Custom LoRA Support** — Load LoRA weights from HTTPS URLs, HuggingFace Hub, or local `.safetensors`
- **Multi-LoRA Mixing** — Load multiple LoRA adapters simultaneously with independent per-adapter weights
- **Tunable Scheduler Shift** — `FlowMatchEulerDiscreteScheduler` with configurable shift (default 1.5)
- **2nd Pass Detailer** — Optional non-destructive detail-injection pass that adds sharpness and micro-texture while preserving base image style, colors, and composition
- **Tiled 4× Upscaler** — Optional DRCT-L super-resolution via [4xRealWebPhoto_v4_drct-l](https://github.com/Phhofm/models) with feather-blended 384px tiles and 64px overlap
- **Flexible Generation** — Configurable resolution, steps, guidance scale, shift, and batch size
- **LoRA Stack Switching** — When the requested LoRA set changes between requests, the pipeline reinitializes to safely load new adapters before CPU offload hooks are attached
- **Multiple Output Formats** — PNG, JPEG, WebP support
- **S3 Storage** — Upload images to S3 with presigned URLs (no base64 size limits)
- **RunPod Model Cache** — HuggingFace cache on network volumes for fast restarts; upscaler model auto-downloaded and cached on first use
- **High-Performance Downloads** — `hf_transfer` backend + Xet storage with concurrent chunk downloads

## 🏗️ Architecture

### Component Loading

Components are loaded individually (ComfyUI-style) to minimize peak VRAM:

| Component | Source | dtype | VRAM |
|-----------|--------|-------|------|
| Transformer | `black-forest-labs/FLUX.2-klein-9b-fp8` → `flux-2-klein-9b-fp8.safetensors` | `float8_e4m3fn` | ~4.5 GB |
| Text Encoder | `edicamargo/qwen_3_8b_fp8mixed_abliterated` weights + `Qwen/Qwen3-8B-FP8` config | `float8_e4m3fn` | ~8 GB |
| VAE | `black-forest-labs/FLUX.2-klein-9B`, subfolder `vae` | `bfloat16` | ~0.7 GB |
| Tokenizer | `Qwen/Qwen3-8B-FP8` | — | — |
| Scheduler | `black-forest-labs/FLUX.2-klein-9B`, subfolder `scheduler` | — | — |

The text encoder (abliterated) has no `config.json` in its repo — it is initialized on meta device using the official `Qwen/Qwen3-8B-FP8` config, then the abliterated state dict is loaded in-place via `load_state_dict(..., assign=True)`. Architecture is identical to the official model; only the refusal vectors are removed.

The transformer and text encoder are downloaded as single safetensors files to `/runpod-volume/models/flux2-klein/`. The VAE, tokenizer, and scheduler are fetched via `from_pretrained` and cached under `HF_HOME` (`/runpod-volume/huggingface`).

### Data Flow

1. **Input Validation** — Parameters validated against schema, preset defaults applied
2. **Pipeline Init** — Components loaded individually; **LoRA adapters loaded here** (before CPU offload hooks); `enable_model_cpu_offload()` called; VAE slicing/tiling enabled
3. **LoRA Stack Check** — If requested adapter stack differs from loaded, pipeline reinitializes
4. **Scheduler Update** — `FlowMatchEulerDiscreteScheduler.from_config(config, shift=N)` applied per request
5. **Image Generation** — Flow-match inference via `Flux2KleinPipeline`
6. **2nd Pass Detailer** *(optional)* — Pipeline called again with 1st pass image as `image` conditioning at low steps; high-frequency detail blended back onto 1st pass result
7. **Tiled Upscale** *(optional)* — DRCT-L 4× SR via spandrel with 384px/64px feather-blended tiles; result resized to `upscale_factor`
8. **Output** — Images uploaded to S3 (presigned URL) or encoded to base64

### LoRA Loading Order (Critical)

LoRA adapters **must** be loaded before `enable_model_cpu_offload()`. PEFT adapter registration conflicts with accelerate's offload hooks if done after. When the LoRA stack changes between requests, the full pipeline reinitializes to preserve this ordering.

## 🚀 Quick Start

### Prerequisites

- **Docker** — For containerized deployment
- **RunPod Account** — For serverless GPU deployment
- **NVIDIA GPU** — RTX 4090 (24 GB) or better; H100 / A100 80GB recommended
- **HuggingFace Token** — Required for `black-forest-labs/FLUX.2-klein-9b-fp8` (gated) and `black-forest-labs/FLUX.2-klein-9B` VAE/scheduler (gated)

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

1. **Push code to GitHub:**
```bash
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. **Create Serverless Template** at [console.runpod.io/serverless/user/templates](https://console.runpod.io/serverless/user/templates):
   - Source: **GitHub**
   - **Container Disk**: 30 GB
   - **Min/Max Workers**: 0–10
   - **GPU**: NVIDIA H100 or RTX 4090
   - **Network Volume**: Recommended (30 GB+) for model caching

3. **Set Environment Variables** (see Configuration section below)

4. **Deploy** — RunPod builds and deploys automatically. Future pushes trigger redeployment.

## 📖 Documentation

### Optimized Presets

All presets are tuned for the **distilled** FLUX.2-klein-9B model (4–16 steps, guidance 1.0).

#### General Purpose Presets

| Preset | Steps | Guidance | Shift | Resolution | Best For |
|--------|-------|----------|-------|------------|----------|
| `realistic_character` | 8 | 1.0 | 1.5 | 1024×1024 | **Photorealistic human LoRAs** (default) |
| `portrait_hd` | 8 | 1.0 | 1.5 | 1024×1536 | High-detail vertical portraits |
| `cinematic_full` | 8 | 1.0 | 1.5 | 1536×1024 | Full-body cinematic compositions |
| `fast_preview` | 4 | 1.0 | 1.5 | 1024×1024 | Quick prompt/seed testing |
| `maximum_quality` | 16 | 1.0 | 1.0 | 1024×1024 | Highest quality final output |

#### Character LoRA Optimized Presets

| Preset | Steps | Guidance | Shift | Resolution | Best For |
|--------|-------|----------|-------|------------|----------|
| `character_portrait_best` | 12 | 1.0 | 2.5 | 1024×1024 | **Best quality for character portraits** — natural skin, maximum fidelity |
| `character_portrait_vertical` | 12 | 1.0 | 2.0 | 896×1152 | Head/shoulders portraits — 4:5 ratio ideal for faces |
| `character_cinematic` | 8 | 1.0 | 2.5 | 1344×896 | Full-body or environmental shots — cinematic 3:2 horizontal |
| `manga_style` | 8 | 1.0 | 1.5 | 1024×1024 | Stylized / manga / illustration output |

### Distilled Model Best Practices

> **This is the distilled (step + guidance distilled) model.** Do not use BASE model settings (35–50 steps, guidance 2.0+).

**Optimal Inference Settings:**
- **Steps:** 4–16 (sweet spot: 8–12 for quality/speed balance; 4 for real-time previewing)
- **Guidance:** 1.0 — the model is guidance-distilled; this is the optimal value. Values above ~2.0 cause over-saturation and artifacts
- **Scheduler Shift:** 1.0–1.5 for photorealism; 2.0–2.5 for strong character identity with LoRAs
- **Resolution:** 1024×1024 standard, 1024×1536 for portraits
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
- Avoid SDXL boilerplate ("masterpiece, best quality", "photorealistic, highly detailed skin texture") — these are associated with over-processed aesthetics
- Use camera/film language instead: "ISO 800", "slight film grain", "unretouched", "candid"
- FLUX.2-klein does **not** support `negative_prompt` — it is ignored

### Scheduler Shift Reference

| Shift | Effect |
|-------|--------|
| `1.0` | Maximum fine-detail budget; most natural skin/texture; best for close-up portraits |
| `1.5` | **Default**; natural texture with good structure |
| `2.0` | Balanced; slightly stronger large-structure coherence |
| `2.5` | BFL recommendation for character LoRAs; excellent identity preservation |
| `3.0` | Better face/structure coherence; may smooth skin texture |
| `5.0` | Strong structure emphasis; helps complex lighting or multi-figure scenes |

### API Reference

**Request Format:**
```json
{
  "input": {
    "prompt": "TOK woman, close-up portrait, soft window light, Sony A7IV, 105mm f/2.8, natural skin, visible pores, unretouched, ISO 800",
    "preset": "character_portrait_best",
    "loras": [
      {
        "path": "https://example.com/your-character-lora.safetensors",
        "scale": 0.85,
        "adapter_name": "character"
      },
      {
        "path": "https://example.com/your-style-lora.safetensors",
        "scale": 0.45,
        "adapter_name": "style"
      }
    ],
    "lora_scale_mode": "normalized",
    "seed": 42,
    "return_type": "s3",
    "enable_2nd_pass": true,
    "second_pass_strength": 0.2,
    "second_pass_steps": 4,
    "second_pass_guidance_scale": 1.0,
    "second_pass_lora_scale_multiplier": 0.7,
    "enable_upscale": true,
    "upscale_factor": 2.0,
    "upscale_blend": 0.35
  }
}
```

**Legacy single-LoRA input is still supported:**
```json
{
  "input": {
    "prompt": "TOK woman, close-up portrait, soft window light",
    "lora_path": "https://example.com/your-character-lora.safetensors",
    "lora_scale": 0.85
  }
}
```

**Response Format (S3):**
```json
{
  "image_urls": ["https://s3.amazonaws.com/bucket/flux2-klein/uuid.jpg?X-Amz-Algorithm=..."],
  "metadata": {
    "loras": [{"adapter_name": "character", "effective_scale": 0.85}],
    "generation_time": "3.21s"
  }
}
```

**Response Format (Base64 fallback):**
```json
{
  "images": ["<base64_encoded_image>"],
  "metadata": {
    "loras": [],
    "generation_time": "3.21s"
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
| `num_inference_steps` | int | `8` | Denoising steps — 4 for speed, 8–12 for quality, 16 for maximum |
| `guidance_scale` | float | `1.0` | CFG scale — keep at 1.0 for distilled model; values above ~2.0 cause artifacts |
| `shift` | float | `1.5` | Scheduler shift — 1.0–1.5 for natural skin; 2.0–2.5 for character identity |
| `seed` | int | `-1` | Random seed (-1 for random) |
| `num_images` | int | `1` | Number of images per request |
| `output_format` | string | `"jpeg"` | Output format: `png`, `jpeg`, `webp` |
| `return_type` | string | `"s3"` | Response type: `s3` (presigned URL) or `base64` |
| `max_sequence_length` | int | `512` | Maximum token length for text encoding |
| `loras` | list | `[]` | Multi-LoRA stack. Each item: `{ "path": string, "scale": float, "adapter_name": string? }`. Path aliases: `url`, `lora_url`, `lora_path`. Scale aliases: `strength`, `weight`, `lora_scale` |
| `lora_scale_mode` | string | `"absolute"` | `absolute` (raw scales) or `normalized` (scales normalized to sum 1.0) |
| `lora_path` | string | `""` | Legacy single-LoRA path; alias: `lora_url` |
| `lora_scale` | float | `0.85` | Legacy single-LoRA scale |
| `additional_lora` | string | `""` | Second LoRA path (legacy); aliases: `additional_lora_path`, `additional_lora_url`, `addition_lora`, `addition_lora_url` |
| `additional_lora_strength` | float | `0.85` | Second LoRA scale (legacy); alias: `additional_lora_scale` |
| **2nd Pass Detailer** | | | |
| `enable_2nd_pass` | bool | `false` | Enable low-denoise img2img detailer pass |
| `second_pass_strength` | float | `0.2` | Denoise strength — 0.2 adds skin/pore detail without composition drift |
| `second_pass_steps` | int | `4` | Steps for 2nd pass |
| `second_pass_guidance_scale` | float | `1.0` | CFG for 2nd pass — keep at 1.0 |
| `second_pass_lora_scale_multiplier` | float | `1.0` | Multiplier for LoRA scales in 2nd pass |
| **Upscaler** | | | |
| `enable_upscale` | bool | `false` | Enable DRCT-L 4× tiled super-resolution |
| `upscale_factor` | float | `2.0` | Target upscale multiplier; model runs at 4×, result resized to this factor |
| `upscale_blend` | float | `0.35` | Blend of AI upscale over LANCZOS (0.0–1.0); 0.35 preserves color/composition |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `HF_TOKEN` | `""` | **Required** — gated access to `black-forest-labs/FLUX.2-klein-9b-fp8` and `black-forest-labs/FLUX.2-klein-9B` |
| `MODEL_ID` | `black-forest-labs/FLUX.2-klein-9B` | VAE/scheduler source repo |
| `DEVICE` | `cuda` | Compute device |
| `UPSCALER_MODEL_PATH` | `/runpod-volume/models/4xRealWebPhoto_v4_drct-l.pth` | DRCT-L upscaler path; auto-downloaded on first use |

### S3 Configuration (Optional)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `S3_BUCKET_NAME` | `""` | S3 bucket for image storage |
| `S3_REGION` | `us-east-1` | AWS region |
| `S3_ACCESS_KEY_ID` | `""` | AWS access key |
| `S3_SECRET_ACCESS_KEY` | `""` | AWS secret key |
| `S3_ENDPOINT_URL` | `""` | Custom S3 endpoint (e.g., MinIO, Wasabi) |
| `S3_PRESIGNED_URL_EXPIRY` | `3600` | Presigned URL expiry in seconds |

When `S3_BUCKET_NAME` is set, images are uploaded to S3 and a presigned URL is returned instead of base64, avoiding payload size limits.

### HuggingFace Performance Tuning

| Environment Variable | Value | Description |
|---------------------|-------|-------------|
| `HF_HOME` | `/runpod-volume/huggingface` | HuggingFace cache directory (network volume) |
| `HF_HUB_CACHE` | `/runpod-volume/huggingface/hub` | Model cache directory |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable `hf_transfer` for fast parallel downloads |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Enable high-performance Xet downloads |
| `HF_XET_NUM_CONCURRENT_RANGE_GETS` | `32` | Concurrent download chunks (default: 16) |
| `HF_HUB_ETAG_TIMEOUT` | `30` | Timeout for metadata requests (seconds) |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `300` | Timeout for file downloads (seconds) |

### VRAM Requirements

| Configuration | Peak VRAM | Notes |
|---------------|-----------|-------|
| FP8 components + cpu offload | ~14–16 GB | **Default** — components swapped CPU↔GPU per forward pass |
| FP8 components, no offload | ~14 GB loaded simultaneously | Faster; requires 16 GB+ free VRAM |
| FP8 + 2nd pass | ~14–16 GB | Same pipeline reused — zero additional VRAM |
| FP8 + upscaler | ~14–16 GB + ~300 MB | DRCT-L tiled; per-tile VRAM negligible |

Component breakdown: transformer FP8 ~4.5 GB + Qwen3-8B FP8 ~8 GB + VAE bf16 ~0.7 GB = ~13.2 GB loaded. With `enable_model_cpu_offload()` peak is lower as only the active component is on GPU at any time.

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

**Training → Inference Alignment:** Train against the base model (`FLUX.2-klein-base-9B`) for maximum LoRA flexibility. Inference on the distilled model with the trained LoRA works well at 8–12 steps, guidance 1.0, shift 2.5.

## 📝 Prompt Engineering Guide

See [SYSTEMPROMPT.md](SYSTEMPROMPT.md) for a comprehensive system prompt template that converts user input into optimized FLUX.2-klein prompts. Includes prompt structure framework, camera/lens references, anti-artifact keyword list, and implementation examples.

## 📄 License

This code is provided as-is for RunPod serverless deployments. The FLUX.2-klein-9B model is licensed under the [FLUX Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md). The 4B models are Apache 2.0.

## 🙏 Acknowledgments

- **[ai-toolkit](https://github.com/ostris/ai-toolkit)** by ostris — LoRA training framework
- **[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)** by Black Forest Labs — distilled model
- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-FP8)** by Alibaba — text encoder
- **[RunPod](https://runpod.io)** — Serverless GPU infrastructure

## 📧 Support

- **This deployment:** Check logs in RunPod console
- **Model behavior:** Consult [FLUX.2-klein model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- **Training:** See [ai-toolkit issues](https://github.com/ostris/ai-toolkit/issues)
