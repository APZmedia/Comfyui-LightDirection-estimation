# Estimating Light Direction and Hardness from Normal Passes and Luma Deltas

This document describes the **theory** behind using **normal passes** (from OpenGL/DirectX rendering) combined with **luma differences** between two frames (before/after) to estimate:

1. Whether the dominant light comes from the **left or right** (and optionally up/down).
2. Whether the light is **hard or soft**.

---

## 1. Inputs

- **Normal Pass** (`N_rgb`):  
  Encoded in RGB channels with X=R, Y=G, Z=B.  
  Values in [0, 1] mapped from [-1, 1].  
  - OpenGL → Y is **up**  
  - DirectX → Y is **down** (requires flipping).

- **Luma Before/After** (`Y_before`, `Y_after`):  
  Grayscale images representing brightness in two states.

---

## 2. Decoding Normals

To decode:

```
Nx = R * 2 - 1
Ny = G * 2 - 1
Nz = B * 2 - 1

# Flip Y if using DirectX convention
Ny = -Ny
```

---

## 3. Computing Luma Deltas

We compare luma frames to detect lighting changes:

```
ΔY = Y_after - Y_before
```

Optionally normalize by exposure drift:

```
scale = median(|ΔY|) * 6
ΔYn = clip(ΔY / scale, -3, 3)
```

---

## 4. Estimating Light Direction

We fit a linear model:

```
ΔY ≈ a * Nx + b * Ny + c
```

- `a` → influence of left/right
- `b` → influence of up/down

The estimated light direction (screen space) is:

```
L = normalize([a, b])
```

- If `L.x > 0` → light from **right**
- If `L.x < 0` → light from **left**

Confidence can be estimated by comparing variance explained by the model vs residuals.

---

## 5. Estimating Hard vs Soft Light

### Approach A – High-Frequency Energy
Hard light yields sharper shadows → more high-frequency energy in ΔY.

- Compute gradient magnitude of ΔY (Sobel filter).  
- Ratio of gradient energy vs signal amplitude gives hardness.

### Approach B – Penumbra Width
Measure transition width of intensity changes along light direction.

```
penumbra ≈ ΔY / gradient_along_light
```

- Larger penumbra width → softer light.

### Combined Estimate
Blend the two measures for robustness:

```
softness = 0.5 * softness_highfreq + 0.5 * softness_penumbra
hardness = 1 - softness
```

---

## 6. Summary of Outputs

- **Light Direction (2D vector)** – `[x, y]`
- **Main Direction** – left/right (from sign of x)
- **Softness** – [0, 1], where 0 = hard, 1 = soft
- **Confidence** – ratio of explained variance to residuals

---

## 7. Recommended Libraries

To implement this workflow in Python or node-based systems:

- **NumPy** → efficient array math and least-squares fitting
- **OpenCV (cv2)** → Sobel gradients, Gaussian blur, morphological ops
- **SciPy** → signal processing, advanced regression tools
- **PyTorch** or **TensorFlow** (optional) → GPU acceleration for large frames
- **ComfyUI Custom Nodes** → integrate into node-based AI/graphics workflows

---

## 8. Notes for Integration

- Always check if the normal pass uses **OpenGL (Y up)** or **DirectX (Y down)** before interpreting.  
- Preprocess luma to reduce exposure drift (normalization).  
- Use masks (e.g., Nz > 0, ΔY above threshold) to ignore irrelevant regions.  
- Smooth results over time for video sequences to stabilize estimates.

---

## 📚 References

- OpenGL and DirectX coordinate systems
- Lambertian reflectance model
- Image processing techniques for edge detection and frequency analysis

