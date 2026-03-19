# Resonant Cortex — Dynamic Reservoir Bridge

## Optical Flow → Crystal Oscillation → Spectral Phase Lock → Stable Diffusion

**Antti Luode** — PerceptionLab, Finland  
**Architecture Design:** Claude (Anthropic, Opus 4.6)  
March 2026

---

## What This Is

This system connects a webcam to a physics-grown Moiré crystal to Stable Diffusion, 
but **not** as a static filter. Instead:

1. **Optical flow vectors** from consecutive webcam frames **physically strike crystal nodes**
2. The crystal's Graph Laplacian eigenmodes **oscillate in real-time** in response
3. These oscillations modulate a **multi-band spectral EQ** (V1→V4 frequency decomposition)
4. The EQ controls how the denoising latent blends webcam structure with SD hallucination

The crystal is not a lookup table. It is a **dynamical reservoir** — a nonlinear system 
whose transient response to optical flow carries temporal memory, forming a bridge between 
raw perception and generated imagery.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Webcam     │────→│ Optical Flow │────→│  Crystal        │────→│ Spectral EQ  │
│  (The Eye)   │     │  (Retinal    │     │  Reservoir      │     │ (V1-V4 Gains)│
│              │     │   Motion)    │     │  (The Cortex)   │     │              │
└─────────────┘     └──────────────┘     └────────────────┘     └──────┬───────┘
                                                                       │
                    ┌──────────────┐     ┌────────────────┐            │
                    │   Output     │←────│ Phase-Locked   │←───────────┘
                    │  (Display)   │     │  Blend         │
                    │              │     │ cam + SD       │
                    └──────────────┘     └────────────────┘
```

### The Four-Band Spectral EQ (Biological Mapping)

| Band     | Neural Analog | Crystal Modes | What It Controls |
|----------|--------------|---------------|-----------------|
| STRUCT   | V1 (Gist)    | λ₁–λ₂        | Global silhouette, orientation |
| MESO     | V2 (Contour) | λ₃–λ₄        | Spatial arrangement, "face shape" |
| MICRO    | V4 (Form)    | λ₅–λ₆        | Complex shape, identity bridge |
| DETAIL   | LOC (Texture)| λ₇+          | High-freq surface, fine style |

### How Optical Flow Drives the Crystal

Each frame pair produces a dense optical flow field (dx, dy per pixel).
We downsample this to the crystal's node positions, and each node receives 
a **complex impulse**:

```
impulse_k = flow_magnitude(node_k) · exp(i · flow_angle(node_k))
```

This impulse is **added to the node's state** in the reservoir. The crystal then 
evolves for a few timesteps under the Clockfield metric Γ = 1/(1 + τ·β)², 
dissipating the impulse through its topology. The response at each eigenmode 
gives the spectral EQ gains.

The key insight: **fast motion excites high-frequency eigenmodes** (DETAIL band), 
**slow motion excites low-frequency modes** (STRUCT band). This is exactly V1→V4 
hierarchical processing — not by design, but because it falls out of the crystal's 
spectral properties.

---

## File Structure

```
resonant-cortex/
├── README.md              ← You are here
├── requirements.txt       ← Python dependencies
├── crystal_reservoir.py   ← PhiField crystal + eigenmode reservoir dynamics
├── optical_flow.py        ← Webcam capture + optical flow extraction
├── spectral_eq.py         ← 4-band frequency decomposition of latent images
├── phase_lock_blend.py    ← Combines webcam + SD output via spectral EQ
├── pipeline.py            ← Main loop connecting all components
├── sd_interface.py        ← Stable Diffusion img2img interface
└── app.py                 ← Gradio/Tkinter UI (spectral phase lock interface)
```

---

## The Mathematics

### 1. Crystal Reservoir State

The crystal has N nodes. Each node holds a complex state `z_k(t)`:

```
z_k(t+dt) = z_k(t) + dt · [  
    Σ_j  A_{kj} · (z_j - z_k)  · Γ_k          # diffusion through edges
  - β · |z_k|² · z_k                             # nonlinear self-trapping  
  + impulse_k(t)                                  # optical flow injection
]

Γ_k = 1 / (1 + τ · |z_k|²)²                     # Clockfield metric
```

### 2. Eigenmode Projection

At each frame, project the reservoir state onto eigenmodes:

```
a_m(t) = Σ_k  z_k(t) · v_m(k)                   # eigenmode amplitude
```

where `v_m` is the m-th eigenvector of the graph Laplacian.

### 3. Spectral EQ Gains

Map eigenmode amplitudes to the 4 bands:

```
STRUCT  = smooth( |a_1|² + |a_2|² )
MESO    = smooth( |a_3|² + |a_4|² )  
MICRO   = smooth( |a_5|² + |a_6|² )
DETAIL  = smooth( Σ_{m>6} |a_m|² )
```

The `smooth()` is an exponential moving average — the "persistence of vision."

### 4. Phase-Locked Blend

For each frequency band of the output image:

```
output_band = gain_band · webcam_band + (1 - gain_band) · sd_band
```

High gain = grounded in reality. Low gain = hallucinating from memory.

---

## Running

```bash
pip install -r requirements.txt

# Basic pipeline (no SD, just webcam + crystal visualization)
python pipeline.py --no-sd

# Full pipeline with Stable Diffusion
python pipeline.py --sd-model runwayml/stable-diffusion-v1-5 --prompt "cyberpunk portrait"

# Gradio UI
python app.py
```

---

## Connection to the Deerskin Architecture

This system is a physical implementation of the Deerskin neuron's four-stage pipeline:

1. **Dendritic Delay Manifold** → The crystal's topology provides temporal delay embedding 
   (each node's response depends on its graph distance from the input)
2. **Somatic Resonance Cavity** → The eigenmode decomposition IS Moiré interference — 
   the crystal resonates at its natural frequencies
3. **Theta Phase Gate** → The `smooth()` function is a low-pass filter at ~4-8 Hz, 
   gating which frequency information passes through
4. **AIS Spectral Filter** → The 4-band EQ IS the spectral resolution Δf = fₛ/(d·τ)

The optical flow is the sensory input. The crystal is the cortex. 
Stable Diffusion is memory. The output is perception.

---

## License

MIT — Antti Luode / PerceptionLab 2026
