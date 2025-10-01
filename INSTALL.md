# Installation Guide

## Quick Installation

1. **Copy to ComfyUI custom nodes directory:**
   ```
   ComfyUI/custom_nodes/Comfyui-LightDirection-estimation/
   ```

2. **Restart ComfyUI**

3. **Find the node** under "Custom/Lighting" category

## Manual Installation

If auto-installation fails:

```bash
cd ComfyUI/custom_nodes/Comfyui-LightDirection-estimation/
pip install -r requirements.txt
```

## Development Installation

For development purposes:

```bash
git clone https://github.com/your-username/Comfyui-LightDirection-estimation.git
cd Comfyui-LightDirection-estimation
pip install -e .
```

## Dependencies

- torch>=1.9.0
- numpy>=1.21.0
- Pillow>=8.0.0
- scipy>=1.7.0

## Troubleshooting

### Common Issues

**"Module not found" errors**
- Ensure all dependencies are installed
- Check Python path includes the custom_nodes directory

**"Node not appearing in ComfyUI"**
- Verify the folder structure is correct
- Check that `__init__.py` files are present
- Restart ComfyUI completely

**"Import errors"**
- Check that all required packages are installed
- Verify Python version compatibility (3.8+)

### Manual Dependency Installation

```bash
pip install torch>=1.9.0 numpy>=1.21.0 Pillow>=8.0.0 scipy>=1.7.0
```

## Verification

After installation, you should see:
- "Normal Map Light Estimator" node in ComfyUI under "Custom/Lighting"
- No import errors in ComfyUI console
- All dependencies properly installed
