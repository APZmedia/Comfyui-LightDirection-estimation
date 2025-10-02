
# What ComfyUI expects (core conventions)

* **IMAGE** → `torch.Tensor` shaped **`[B, H, W, C]`** with **`C = 3`** (RGB, *channel-last*). ([ComfyUI Documentation][1])
* **MASK** → `torch.Tensor` shaped **`[B, H, W]`** with float values typically in **`[0, 1]`**. (Single-channel; batch first, no explicit `C`.) ([ComfyUI Documentation][1])
* **LATENT** → `dict` with key `"samples"` as a tensor shaped **`[B, C, H, W]`** with **`C = 4`** (*channel-first*). Not directly relevant for PIL I/O, but useful context. ([ComfyUI Documentation][1])
* **Return tuples**: if your node returns a single tensor (e.g., one IMAGE), return **`(image,)`**, not `image`. ([ComfyUI Documentation][1])
* **Batch rules & shapes**: ComfyUI images are almost always batches; be mindful of **squeeze/unsqueeze** when you receive or emit single items. ([ComfyUI Documentation][2])
* **Mask creation in `LoadImage`**: by default, ComfyUI builds a MASK from the **alpha channel**, **normalizes to `[0,1]`**, then **inverts** (so fully opaque alpha → mask value 0). If there’s no alpha, it creates a default mask `[1, 64, 64]`. Mirror this behavior if you mimic `LoadImage`. ([ComfyUI Documentation][1])
* **Authoritative “getting started” for custom nodes**: structure, `INPUT_TYPES`, `RETURN_TYPES`, etc. ([ComfyUI Documentation][3])

# Minimal, correct conversion helpers

These helpers follow the **official shapes** above and are safe defaults for custom nodes that need to hop between PIL and Comfy tensors.

```python
# --- IMAGE: PIL <-> Tensor([B,H,W,3]) ---

import torch, numpy as np
from PIL import Image, ImageOps

def pil_to_comfy_image_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Returns a float32 tensor in [0,1] shaped [1, H, W, 3] (RGB, NHWC).
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.asarray(pil_img, dtype=np.uint8)  # [H, W, 3], 0..255
    t = torch.from_numpy(arr).to(torch.float32) / 255.0         # [H, W, 3], 0..1
    t = t.unsqueeze(0).contiguous()                             # [1, H, W, 3]
    return t

def comfy_image_tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Accepts [B,H,W,3] in [0,1]. Takes the first item if B>1.
    """
    assert img.ndim == 4 and img.shape[-1] == 3, "Expected [B,H,W,3]"
    t = img[0].clamp(0, 1).detach().cpu()                       # [H, W, 3]
    arr = (t.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
```

```python
# --- MASK: PIL (L or A) <-> Tensor([B,H,W]) ---

def pil_alpha_to_comfy_mask_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Mirrors LoadImage: take alpha, normalize to [0,1], then invert -> [1,H,W].
    If no alpha, returns default [1,64,64] zeros (opaque -> mask 0).
    """
    if "A" in pil_img.getbands():
        alpha = np.asarray(pil_img.getchannel("A"), dtype=np.float32)  # [H,W], 0..255
        mask = 1.0 - (alpha / 255.0)                                   # invert
        t = torch.from_numpy(mask).unsqueeze(0).contiguous()           # [1,H,W]
        return t
    else:
        return torch.zeros((1, 64, 64), dtype=torch.float32)           # default

def pil_luma_to_comfy_mask_tensor(pil_mask: Image.Image) -> torch.Tensor:
    """
    For a grayscale mask image: normalize to [0,1], no inversion.
    """
    if pil_mask.mode != "L":
        pil_mask = pil_mask.convert("L")
    m = np.asarray(pil_mask, dtype=np.float32) / 255.0                 # [H,W]
    return torch.from_numpy(m).unsqueeze(0).contiguous()               # [1,H,W]

def comfy_mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    """
    Accepts [B,H,W] in [0,1]; exports first mask as 8-bit 'L' PIL image.
    """
    assert mask.ndim in (2,3), "Expected [H,W] or [B,H,W]"
    m = mask[0] if mask.ndim == 3 else mask
    m = m.clamp(0,1).detach().cpu().numpy()
    return Image.fromarray((m * 255.0).round().astype(np.uint8), mode="L")
```

### Why these shapes & steps?

* **Channel-last images `[B,H,W,3]`** and **3-D masks `[B,H,W]`** are straight from the official Comfy docs for backend datatypes. The docs also call out that PyTorch ops may be channel-first, so convert explicitly when needed. ([ComfyUI Documentation][1])
* The **“return `(image,)`”** detail prevents subtle output packing bugs in nodes with single outputs. ([ComfyUI Documentation][1])
* **Squeeze/unsqueeze** gotchas are common when a batch size is 1; the docs emphasize this. ([ComfyUI Documentation][2])
* The **LoadImage alpha→mask (normalized then inverted)** is documented, so matching it gives users consistent behavior when your node “acts like” loading transparency as a mask. ([ComfyUI Documentation][1])

# Skeleton node patterns (LLM-friendly)

### 1) A node that accepts an IMAGE and emits an IMAGE

```python
class MyImageNode:
    CATEGORY = "example/utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, images):
        # images: torch.Tensor [B,H,W,3] in [0,1]
        # Example: no-op passthrough
        out = images  # ensure shape and dtype unchanged
        return (out,)  # important: tuple!
```

(Comfy’s “Getting Started” shows the basic node shape and `INPUT_TYPES`/`RETURN_TYPES` conventions.) ([ComfyUI Documentation][3])

### 2) A node that takes an IMAGE + MASK and composites them

```python
class ApplyMaskToImage:
    CATEGORY = "example/masks"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, images, mask):
        # images: [B,H,W,3] in [0,1]
        # mask:   [B,H,W]   in [0,1]  -> unsqueeze channel to broadcast
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)                  # [1,H,W]
        m = mask.unsqueeze(-1)                        # [B,H,W,1]
        out = images * (1.0 - m)                      # simple “erase” where mask=1
        return (out,)
```

(Using `unsqueeze` this way is explicitly recommended when aligning MASK shapes with IMAGE shapes.) ([ComfyUI Documentation][1])

### 3) A node that converts IMAGE → PIL → IMAGE (e.g., to use a PIL-only op)

```python
class PILFilterExample:
    CATEGORY = "example/pil"
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    def run(self, images):
        # Take first from batch, convert to PIL
        pil = comfy_image_tensor_to_pil(images)

        # Do any PIL operations you need:
        pil = ImageOps.autocontrast(pil)

        # Back to Comfy tensor (still batch size 1)
        out = pil_to_comfy_image_tensor(pil)
        return (out,)
```

# Practical tips for robust custom nodes

* **Be explicit about device transfers**: do CPU conversions before PIL/Numpy (`.detach().cpu()`), and only move back to GPU when needed. (PIL/Numpy work on CPU.)
* **Clamp and dtype**: before `uint8` conversion, **`clamp(0,1)`**, then multiply by 255 and cast — prevents wraparound artifacts.
* **Use `.contiguous()`** after reshapes/unsqueezes to avoid stride surprises in downstream ops.
* **Check shapes up front**: defensive `assert`s or `if` branches on `img.ndim` / `mask.ndim` eliminate many runtime errors.
* **Remember channel order**: IMAGE is NHWC; many PyTorch models expect NCHW → use `permute(0,3,1,2)` and the inverse when necessary.
* **Batch semantics**: Even “single” images are batches of size 1 — keep it that way unless you have a reason to squeeze. ([ComfyUI Documentation][2])

# Sources (official docs)

* **Images, Latents, and Masks** — shapes, mask behavior from `LoadImage`, NHWC vs NCHW notes. ([ComfyUI Documentation][1])
* **Working with `torch.Tensor`** — batch semantics, squeeze/unsqueeze guidance. ([ComfyUI Documentation][2])
* **Custom node walkthrough** — node structure, inputs/outputs, and examples. ([ComfyUI Documentation][3])

If you want, I can turn these helpers into a tiny **`utils.py`** and a ready-to-publish **node pack scaffold** (with `pyproject.toml`, `NODE_CLASS_MAPPINGS`, etc.) so you can drop it into `custom_nodes/` and start iterating.

[1]: https://docs.comfy.org/custom-nodes/backend/images_and_masks "Images, Latents, and Masks - ComfyUI"
[2]: https://docs.comfy.org/custom-nodes/backend/tensors "Working with torch.Tensor - ComfyUI"
[3]: https://docs.comfy.org/custom-nodes/walkthrough "Getting Started - ComfyUI"
