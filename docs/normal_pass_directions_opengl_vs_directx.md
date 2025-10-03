# Understanding Normal Pass Direction Mapping in OpenGL and DirectX

When encoding surface normals into a **normal pass** (for deferred shading, post-processing, etc.), engines using **OpenGL** and **DirectX** store the **X, Y, and Z components** of the normals into the **RGB channels** of a texture.

Due to differing **coordinate systems**, the interpretation of **Red (X)** and **Green (Y)** channels varies slightly between the two.

---

## 🎨 Color Encoding of Normals

Surface normals are vectors with components in the range `[-1, 1]`. To store them in a texture, we remap them to `[0, 1]` using:

```c
encoded = (normal * 0.5) + 0.5
```

To decode back:

```c
normal = (encoded * 2.0) - 1.0
```

---

## 🧭 Coordinate Systems

| Feature            | OpenGL                     | DirectX                    |
|--------------------|----------------------------|----------------------------|
| Screen Origin      | Bottom-left                | Top-left                   |
| Handedness         | Right-handed               | Left-handed                |
| +X direction       | Right                      | Right                      |
| +Y direction       | **Up**                     | **Down**                   |
| +Z direction       | Out of screen (toward camera) | Into screen (away from camera) |

---

## 🟥 Red Channel – X Direction

**Red channel encodes the X (left-right) direction of the surface normal:**

| Red Value | Direction        |
|-----------|------------------|
| `0.0`     | Strong Left (`X = -1`)  |
| `0.5`     | Center / Forward |
| `1.0`     | Strong Right (`X = +1`) |

✅ This is the **same** in both OpenGL and DirectX.

---

## 🟩 Green Channel – Y Direction

**Green channel encodes the Y (up-down) direction, but differs due to coordinate systems:**

### OpenGL (Origin at bottom-left)
| Green Value | Surface Faces |
|-------------|----------------|
| `0.0`       | Strong **Down** (`Y = -1`) |
| `0.5`       | Horizontal |
| `1.0`       | Strong **Up** (`Y = +1`) |

### DirectX (Origin at top-left)
| Green Value | Surface Faces |
|-------------|----------------|
| `0.0`       | Strong **Up** (`Y = +1`) *(flipped)* |
| `0.5`       | Horizontal |
| `1.0`       | Strong **Down** (`Y = -1`) *(flipped)* |

⚠️ The **Y axis is flipped** between OpenGL and DirectX conventions. This affects how you interpret the **green channel**:
- A surface that appears green in OpenGL because it faces **up**, will appear **dark green (down)** in DirectX unless corrected.
- Many engines flip the green channel of normal maps for compatibility (e.g. Unity vs Unreal).

---

## 🔵 Blue Channel – Z Direction (View Space)

This often represents the depth or “forward” direction of the normal (into or out of the screen):

| Blue Value | Direction |
|------------|-----------|
| `0.0`      | Toward viewer (`Z = -1`) |
| `0.5`      | Tangent to screen |
| `1.0`      | Into screen (`Z = +1`) |

⚠️ This is sometimes flipped depending on whether view space Z is defined as forward or backward in the engine.

---

## ✅ Summary Table

| Channel | Component | Meaning         | OpenGL High Value | DirectX High Value |
|---------|-----------|------------------|-------------------|--------------------|
| Red     | X         | Left–Right       | Right             | Right              |
| Green   | Y         | Up–Down          | Up                | Down *(flipped)*   |
| Blue    | Z         | Forward–Backward | Into screen       | Into screen (depends) |

---

## 🧪 Visual Interpretation Guide

Given a normal encoded in RGB = (1.0, 0.0, 0.5):

- **OpenGL**: Surface normal points **right and down**, flat to screen (Z = 0)
- **DirectX**: Surface normal points **right and up**, flat to screen (Y flipped)

---

## 🔄 Tip for Artists and Developers

If your normal maps look inverted in lighting:

- Try **flipping the green channel** (Y).
- Many tools (Substance, Photoshop, Unity, etc.) have options like:
  - "OpenGL-style normals"
  - "DirectX-style normals"
- This ensures **consistent shading across platforms**.

---

## 📚 References

- Khronos OpenGL Specification
- Microsoft Direct3D Coordinate System
- Unity Documentation on Normal Maps
- Unreal Engine Normal Map Compression
