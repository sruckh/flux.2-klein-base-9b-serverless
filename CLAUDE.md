# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 🛑 STOP — Run codemap before ANY task

```bash
codemap .                     # Project structure
codemap --deps                # How files connect
codemap --diff                # What changed vs main
codemap --diff --ref <branch> # Changes vs specific branch
```

## Project Overview

This is a **RunPod Serverless** deployment for FLUX.2-klein-base-9B image generation with custom LoRA weight support. The implementation is based on [ai-toolkit](https://github.com/ostris/ai-toolkit) and uses flow matching with dynamic shift calculation matching training behavior.

**Primary files:**
- `handler.py` - Main serverless handler (FluxPipeline, presets, validation, generation)
- `Dockerfile` - Container definition (runpod/base, CUDA 12.8, PyTorch 2.8, Flash Attention 2.8.3)
- `requirements.txt` - Python dependencies
- `test_input.json` - Sample API request for testing

## Development Commands

### Local Testing

```bash
# Build Docker image
docker build -t flux-2-klein-serverless .

# Run locally (GPU required)
docker run --gpus all -p 8000:8000 flux-2-klein-serverless

# Test endpoint
curl -X POST http://localhost:8000/runpod/v1/lgpu/input \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

### Deployment

This project uses RunPod's **GitHub integration** for automatic builds. Pushing to the main branch triggers automatic redeployment.

Configure template at: `console.runpod.io/serverless/user/templates`
- **Container Disk**: 30 GB
- **Min/Max Workers**: 0-10 (scale to zero)
- **GPU**: NVIDIA H100 (recommended) or A100 80GB

## Architecture

### Handler Structure

The `handler.py` follows this flow:

1. **Configuration** (lines 32-116): Environment variables, dtype mapping, `PRESETS` dict
2. **Validation** (lines 131-206): `INPUT_SCHEMA` with RunPod's `rp_validator`
3. **Pipeline** (lines 273-329): Global `pipeline` instance with CPU offload, VAE optimizations
4. **Generation** (lines 365-501): Preset application → parameter extraction → `FluxPipeline()` call → base64 encoding

### Key Implementation Details

**Dynamic Shift Calculation** (`calculate_shift()`):
- Matches FLUX training behavior from ai-toolkit
- Formula: `shift = ((max_shift - base_shift) / (max_seq_len - base_seq_len)) * image_seq_len + base_shift`
- `image_seq_len = (height // 2) * (width // 2)` (FLUX patch_size = 2)

**LoRA Loading** (`load_lora_weights()`):
- Supports HuggingFace Hub repo IDs OR local `.safetensors` paths
- Sets adapter scale via `set_adapters()` or `_lora_scale` attribute
- Falls back gracefully on failure (continues without LoRA)

**Presets System**:
- Presets are applied first, then overridden by explicit parameters
- Preset values are copied to avoid mutation
- Default preset: `realistic_character` (28 steps, guidance 2.5, 1024x1024)

### Memory Optimizations

The pipeline uses three optimizations for constrained GPU memory:
```python
pipeline.enable_model_cpu_offload()  # Offload model to CPU between forward passes
pipeline.vae.enable_slicing()         # Process VAE in slices
pipeline.vae.enable_tiling()          # Process VAE in tiles
```

## Optimized Settings Reference

### For Ultra-Realistic Human Character LoRAs

Based on Black Forest Labs documentation and community testing:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Steps | 28-30 | BFL recommends 20-30 for photorealistic styles |
| Guidance | 2.5-3.5 | Lower = more natural/realistic |
| LoRA Scale | 0.8-1.2 | Standard range for character weights |
| Resolution | 1024x1024 or 1024x1536 | Square or portrait (2:3) |

### Training Configuration (for reference)

When training LoRAs with ai-toolkit for realistic characters:
- **Network**: linear=128, linear_alpha=64, conv=64, conv_alpha=32
- **Steps**: 7000 (BFL recommendation for photorealistic)
- **Learning Rate**: 0.000095
- **Weight Decay**: 0.00001
- **Dataset**: 20-40 images with diverse angles/expressions/lighting

## Adding New Presets

Add to the `PRESETS` dict in `handler.py`:

```python
"your_preset_name": {
    "num_inference_steps": N,
    "guidance_scale": X.X,
    "width": W,
    "height": H,
    "max_sequence_length": N,
    "description": "Brief description"
},
```

Then add the preset name to the `INPUT_SCHEMA` constraints:
```python
"preset": {
    ...
    "constraints": lambda x: x in PRESETS or x == "",
},
```

## Common Modifications

**Change default model**: Set `MODEL_ID` environment variable or modify `DEFAULT_MODEL_ID` in handler.py (line 32-34).

**Add new output format**: Update `INPUT_SCHEMA["output_format"]["constraints"]` and `encode_image_to_base64()`.

**Adjust memory usage**: Modify `enable_model_cpu_offload()`, `enable_slicing()`, or `enable_tiling()` calls in `initialize_pipeline()`.

**Support multiple LoRAs**: The current implementation loads one LoRA per request. To support multiple, extend `load_lora_weights()` to accept a list and call `pipeline.load_lora_weights()` iteratively.
