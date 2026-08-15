---
name: ultra-3d-visualizer
description: Create hyper-realistic, volumetric, and deep 3D visual outputs for 3D Gaussian Splatting and WebGL viewers. Use whenever the user requests 3D rendering enhancements, depth-perceptual shading, soft-splat volumetric rendering, studio lighting, or wants outputs to look "very very 3d", photorealistic, or visually immersive.
---

# Ultra 3D Visualizer Skill

A skill for elevating WebGL canvas renderings, 3D point cloud splats, and surface meshes to hyper-realistic, depth-rich 3D visual fidelity.

## Core 3D Rendering Principles

When generating or updating 3D visual viewers (Three.js, WebGL, or Python PyTorch rasterizers):

### 1. Volumetric Gaussian Soft-Splat Shader
- Never render flat, hard-edged square 2D points.
- Use a dynamic circular alpha texture (`createSplatTexture()`) with radial Gaussian decay (`exp(-r^2 * 4)`).
- Enable `transparent: true`, `depthWrite: false`, and `blending: THREE.NormalBlending` or `THREE.AdditiveBlending` for volumetric depth stacking.

### 2. Cinematic 3D Lighting Setup (3-Point Studio)
- **Key Light**: Soft directional light (`0xffffff`, intensity `1.2`) with `castShadow = true` placing crisp contact shadows.
- **Fill Light**: Cool cyan/indigo ambient fill light (`0x00f2fe`, intensity `0.6`) for deep shadow detail.
- **Rim / Back Light**: Warm golden highlight (`0xffaa00`, intensity `0.8`) from behind to pop object silhouettes off the background.

### 3. ACES Filmic Tone Mapping & Shadow Maps
```javascript
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
```

### 4. Ground Contact Shadow & Depth Plane
- Place a shadow-receiving ground plane (`THREE.ShadowMaterial` or subtle radial gradient canvas) under the 3D model at `y = -0.8`.
- Render a spatial perspective grid to provide depth reference cues.

### 5. Tactical PBR Mesh Shading
- For 3D OBJ / GLTF surface meshes, use `THREE.MeshStandardMaterial` or `THREE.MeshPhysicalMaterial` with:
  - `roughness: 0.35`
  - `metalness: 0.15`
  - `clearcoat: 0.2`
  - `vertexColors: true` (or custom palette mapping)
