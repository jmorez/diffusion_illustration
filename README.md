# Diffusion Illustration

A visualization of isotropic and anisotropic diffusion of particles, demonstrating how diffusion causes signal attenuation through dephasing.

## Animation

The animation shows:
- **Left panels**: Isotropic diffusion (equal in all directions)
- **Right panels**: Anisotropic diffusion in a tube with physical barriers
- **Bottom plots**: Signal attenuation S(t) showing diffusion-induced dephasing

![Diffusion Animation](diffusion_animation.mp4)

## Running

```bash
uv run main.py
```

## Requirements

- numpy
- matplotlib
- ffmpeg (for MP4 export)