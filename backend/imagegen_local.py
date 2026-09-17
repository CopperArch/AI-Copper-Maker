#!/usr/bin/env python3
"""Local image generation/editing worker — run inside venv-imagegen, never
imported by the main app process (torch/diffusers are heavy, opt-in-only
deps kept out of the main venv/requirements.txt).

Reads one JSON request from stdin: {"prompt": str, "image_b64": str}
("image_b64" empty/absent -> pure text-to-image; a "data:image/...;base64,..."
data URL -> image-to-image editing). Writes exactly one JSON line to stdout:
{"b64_json": str, "media_type": "image/png"} on success, or {"error": str}
on failure (also exits non-zero).

SDXL-Turbo is the default model: distilled for few-step inference, so it
stays usable across whatever hardware ended up installed (fast on a real
GPU, tolerable on CPU-only, unlike full SDXL/FLUX at 20-50 steps). Image
editing here means img2img (renoise the source toward the prompt) — a real
limitation, not instruction-following editing like "remove the object on
the left"; that needs a dedicated edit-tuned model, which the drop-in
AutoPipeline swap below is deliberately left ready for. diffusers' img2img
only actually runs strength*num_inference_steps denoising steps from the
partially-noised start — confirmed live: strength=0.7 at 4 steps (~2-3 real
steps) was too weak to visibly move the image toward the prompt at all;
8 steps at strength=0.75 (~6 real steps) gives a real, visible edit.
"""
import sys
import json
import base64
import io

MODEL_ID = "stabilityai/sdxl-turbo"


def main():
    req = json.loads(sys.stdin.read())
    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required")
    image_b64 = req.get("image_b64") or ""

    import torch
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    if image_b64:
        from diffusers import AutoPipelineForImage2Image
        _, _, b64data = image_b64.partition(",")
        src = Image.open(io.BytesIO(base64.b64decode(b64data or image_b64))).convert("RGB")
        w, h = src.size
        # SDXL-Turbo wants multiples of 8; cap the long edge at 768 so CPU
        # fallback stays within a sane time budget.
        scale = 768 / max(w, h)
        src = src.resize((max(8, int(w * scale) // 8 * 8), max(8, int(h * scale) // 8 * 8)))
        pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe = pipe.to(device)
        result = pipe(prompt=prompt, image=src, strength=0.75, num_inference_steps=8, guidance_scale=0.0).images[0]
    else:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe = pipe.to(device)
        result = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, width=768, height=768).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print(json.dumps({"b64_json": base64.b64encode(buf.getvalue()).decode(), "media_type": "image/png"}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)
