# FLUX.2-klein-9B RunPod Serverless

RunPod serverless worker for FLUX.2-klein-9B image generation with optional abliterated text-encoder override, LoRA stacking, optional second-pass refinement, optional tiled upscale, and S3/base64 outputs.

## Current Behavior (Source of Truth)

This README reflects the current `handler.py` implementation on `main`.

- Pipeline class: `Flux2KleinPipeline`
- Default model repo: `black-forest-labs/FLUX.2-klein-9B`
- Pipeline load method: full `from_pretrained(...)`
- Text encoder mode:
  - `official` (default): use encoder from model repo
  - `abliterated`: load `edicamargo/qwen_3_8b_fp8mixed_abliterated` safetensors into `pipe.text_encoder`
- LoRA support: load one or multiple adapters, activate with `pipe.set_adapters(...)`, and pass `attention_kwargs={"scale": 1.0}` on every pipeline call so the FLUX attention processor picks up the per-adapter scales
- Scheduler: recreated per request with configurable `shift`
- Distilled guidance handling: requested guidance values above `1.0` are clamped to `1.0` (first pass and second pass)
- Generation context: `torch.no_grad()`
- Output: S3 presigned URL (if configured) or base64

## Architecture

### Initialization Flow

On first request (or when LoRA stack changes):

1. Build `Flux2KleinPipeline` from `MODEL_ID`.
2. If `TEXT_ENCODER_MODE=abliterated`, download and apply abliterated text encoder weights.
3. Load requested LoRA adapters.
4. Activate adapters with effective scales.
5. Enable `pipe.enable_model_cpu_offload()` and VAE slicing/tiling.

### Request-Time Flow

For each request:

1. Validate input against schema.
2. Resolve requested LoRA stack.
3. Reinitialize pipeline if the adapter stack signature changed (`path` + `adapter_name`).
4. Rebuild scheduler with request/preset shift.
5. Re-apply adapter scales via `set_adapters()`.
6. Generate first-pass images (with `attention_kwargs={"scale": 1.0}`).
7. Optional second pass (img2img) with optional LoRA scale multiplier (also uses `attention_kwargs`).
7. Optional tiled upscale.
8. Return S3 URLs or base64.

Second-pass detail mode guardrails:

- `second_pass_steps` is clamped to `1..4`.
- `second_pass_guidance_scale` is forced to `1.0`.
- `second_pass_lora_scale_multiplier` is clamped to `0.0..1.0`.
- Only luminance high-frequency detail is blended back into the base image (color/composition remain from pass 1).

## Input Parameters

### Required

- `prompt` (`str`)

### Common Optional

- `preset` (`str`, default `realistic_character`)
- `width` (`int`, default `1024`)
- `height` (`int`, default `1024`)
- `num_inference_steps` (`int`, default `8`)
- `guidance_scale` (`float`, default `1.0`)
- `shift` (`float`, default `1.5`)
- `seed` (`int`, default `-1` for time-based seed)
- `num_images` (`int`, default `1`)
- `max_sequence_length` (`int`, default `512`)
- `output_format` (`str`, default `jpeg`)
- `return_type` (`str`, schema default `s3`)

### LoRA Parameters

Multi-LoRA (preferred):

- `loras` (`list`, default `[]`)
  - each item supports:
    - `path` (or aliases: `url`, `lora_url`, `lora_path`)
    - `scale` (or aliases: `strength`, `weight`, `lora_scale`) — `0.0` is a valid value and is preserved as-is
    - `adapter_name` (optional; auto-generated if omitted)
    - `weight_name` (optional; useful when loading from HF repo paths)

Legacy single/dual aliases are also supported:

- `lora_path`, `lora_url`, `lora_scale`
- `lora_weight_name`
- `additional_lora`, `additional_lora_path`, `additional_lora_url`
- `addition_lora`, `addition_lora_url`
- `additional_lora_scale`, `additional_lora_strength`
- `addition_lora_scale`, `addition_lora_strength`
- `additional_lora_weight_name`, `addition_lora_weight_name`

LoRA scaling mode:

- `lora_scale_mode`: `absolute` (default) or `normalized`
- `loras` and legacy fields can be combined in one request.
- Adapter names must be unique across all requested LoRAs.

### Second Pass

- `enable_2nd_pass` (`bool`, default `false`)
- `second_pass_strength` (`float`, default `0.2`)
- `second_pass_steps` (`int`, default `4`, runtime clamp `1..4`)
- `second_pass_guidance_scale` (`float`, default `1.0`, runtime forced to `1.0`)
- `second_pass_lora_scale_multiplier` (`float`, default `1.0`, runtime clamp `0.0..1.0`)

### Upscale

- `enable_upscale` (`bool`, default `false`)
- `upscale_factor` (`float`, default `2.0`)
- `upscale_blend` (`float`, default `0.35`)

## Presets

Current presets in code:

- `realistic_character`: 8 steps, guidance 1.0, shift 1.5, 1024x1024
- `portrait_hd`: 8 steps, guidance 1.0, shift 1.5, 1024x1536
- `cinematic_full`: 8 steps, guidance 1.0, shift 1.5, 1536x1024
- `fast_preview`: 4 steps, guidance 1.0, shift 1.5, 1024x1024
- `maximum_quality`: 16 steps, guidance 1.0, shift 1.0, 1024x1024
- `character_portrait_best`: 12 steps, guidance 1.0, shift 2.5, 1024x1024
- `character_portrait_vertical`: 12 steps, guidance 1.0, shift 2.0, 896x1152
- `character_cinematic`: 8 steps, guidance 1.0, shift 2.5, 1344x896
- `manga_style`: 8 steps, guidance 1.0, shift 1.5, 1024x1024

## Distilled Model Notes

- This worker targets the distilled FLUX.2-klein-9B path.
- Guidance values above `1.0` are clamped to `1.0` before pipeline calls (first pass and second pass).
- Response metadata includes `requested_guidance_scale` and `effective_guidance_scale`.
- Second pass is constrained to detail enhancement only (not stylistic re-rendering).

## Environment Variables

### Core

- `MODEL_ID` (default `black-forest-labs/FLUX.2-klein-9B`)
- `HF_TOKEN` (required for gated model access)
- `DEVICE` (default `cuda`)
- `TEXT_ENCODER_MODE` (`official` default, or `abliterated`)

### S3

- `S3_BUCKET_NAME` (empty disables S3)
- `S3_REGION` (default `us-east-1`)
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_ENDPOINT_URL` (optional custom endpoint)
- `S3_PRESIGNED_URL_EXPIRY` (default `3600`)

### Hugging Face Cache (from Dockerfile)

- `HF_HOME=/runpod-volume/huggingface-cache`
- `HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub`

## Output Behavior

- If `return_type == "s3"`, worker attempts S3 upload first.
- If S3 upload fails, worker falls back to base64 per image.
- If fallback base64 payload exceeds size guard (`> ~1.8MB` string), returns an error.
- Metadata includes LoRA application info, generation time, and requested/effective guidance scale.

## LoRA Error Handling

LoRA load failures are fatal for the request:

- Worker raises a runtime error that includes adapter name and source path.
- This avoids silent generation with missing adapters.
- Worker validates that each requested adapter is actually registered on the Flux2 `transformer` before inference.

## Deployment

### Build

```bash
docker build -t flux-2-klein-serverless .
```

### Run locally (GPU required)

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_hf_token \
  -e TEXT_ENCODER_MODE=official \
  flux-2-klein-serverless
```

### RunPod

Use GitHub integration against this repository. Pushes to `main` trigger rebuild/redeploy when configured in RunPod.

## Example Request

Single LoRA:

```json
{
  "input": {
    "prompt": "TOK woman, close-up portrait, soft window light",
    "preset": "portrait_hd",
    "seed": 42,
    "loras": [
      {
        "path": "https://example.com/character.safetensors",
        "scale": 0.85,
        "adapter_name": "character"
      }
    ],
    "enable_2nd_pass": true,
    "second_pass_steps": 4,
    "output_format": "jpeg",
    "return_type": "s3"
  }
}
```

Multiple LoRAs (stacked):

```json
{
  "input": {
    "prompt": "TOK woman, close-up portrait, soft window light",
    "preset": "portrait_hd",
    "seed": 42,
    "loras": [
      {
        "path": "https://example.com/character.safetensors",
        "scale": 0.9,
        "adapter_name": "character"
      },
      {
        "path": "https://example.com/style.safetensors",
        "scale": 0.6,
        "adapter_name": "style"
      }
    ],
    "lora_scale_mode": "absolute",
    "enable_2nd_pass": true,
    "second_pass_steps": 4,
    "second_pass_lora_scale_multiplier": 0.8,
    "enable_upscale": false,
    "output_format": "jpeg",
    "return_type": "s3"
  }
}
```

## Operational Notes

- The worker keeps a global pipeline instance and reloads only when needed (initial load or LoRA stack change).
- Upscaler model (`4xRealWebPhoto_v4_drct-l`) is downloaded lazily on first upscale request.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set to reduce fragmentation risk.

## Troubleshooting

### "Cannot copy out of meta tensor"

Resolved in current code path by avoiding meta-instantiated text encoder replacement. Abliterated weights are applied directly to instantiated `pipe.text_encoder`.

### "Setting requires_grad=True on inference tensor outside InferenceMode"

Resolved by using `torch.no_grad()` in generation paths where adapter switching can occur between passes.

### Second (or additional) LoRA has no visible effect

`Flux2KleinPipeline.__call__()` takes `attention_kwargs` (not `joint_attention_kwargs` — that is an internal transformer parameter). Pass `attention_kwargs={"scale": 1.0}` on every pipeline call for the attention processor to apply the per-adapter scales set by `set_adapters()`. Without it, adapters are loaded and registered but their scale is not picked up during the forward pass. Both pipeline calls in `generate_images` include this kwarg.

## Reference Links

- FLUX.2-klein-9B: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- Diffusers: https://github.com/huggingface/diffusers
- RunPod serverless: https://docs.runpod.io/serverless/overview
