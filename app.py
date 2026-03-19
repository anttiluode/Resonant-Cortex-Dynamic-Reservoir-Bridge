"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RESONANT CORTEX — Gradio App                                                ║
║                                                                              ║
║  Spectral Phase Lock interface with crystal reservoir visualization.          ║
║  For users who don't have a webcam or want to experiment with images.         ║
║                                                                              ║
║  Upload an image → inject as impulse → watch crystal respond →               ║
║  see spectral EQ modulate the image.                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import gradio as gr
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

from crystal_reservoir import grow_crystal, CrystalReservoir
from spectral_eq import SpectralEQ, PhaseLockedBlender
from sd_interface import DummySD

# ═══ Global state ═══
print("Growing crystal reservoir for Gradio app...")
CRYSTAL = grow_crystal(size=48, beta=0.8, steps=300, seed=42, n_modes=12)
BLENDER = PhaseLockedBlender()
DUMMY_SD = DummySD(style="cyberpunk")
EQ = SpectralEQ()


def image_to_impulses(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into crystal node impulses.
    
    Instead of optical flow (which needs two frames), we use the 
    spatial gradient of the image — edges create impulses, flat areas don't.
    This is the "single-frame retina" equivalent.
    """
    if image is None:
        return np.zeros(CRYSTAL.N, dtype=np.complex128)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Compute gradients (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    h, w = gray.shape
    impulses = np.zeros(CRYSTAL.N, dtype=np.complex128)
    
    for k in range(CRYSTAL.N):
        row, col = CRYSTAL.topo.node_positions[k]
        px = int(np.clip(col / CRYSTAL.topo.field_size * w, 0, w - 1))
        py = int(np.clip(row / CRYSTAL.topo.field_size * h, 0, h - 1))
        
        dx = gx[py, px]
        dy = gy[py, px]
        mag = np.sqrt(dx**2 + dy**2) * 2.0
        angle = np.arctan2(dy, dx)
        impulses[k] = mag * np.exp(1j * angle)
    
    return impulses


def plot_crystal_state():
    """Generate crystal visualization."""
    viz = CRYSTAL.get_state_visualization()
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor='#0a0a14')
    
    # 1. Crystal nodes colored by magnitude
    ax = axes[0]
    ax.set_facecolor('#0a0a14')
    mags = viz['node_magnitudes']
    max_mag = max(mags.max(), 0.001)
    
    for k in range(CRYSTAL.N):
        row, col = CRYSTAL.topo.node_positions[k]
        mag_norm = mags[k] / max_mag
        color = plt.cm.inferno(mag_norm)
        size = 20 + mag_norm * 80
        ax.scatter(col, row, s=size, c=[color], edgecolors='#333355', linewidths=0.3)
    
    # Draw edges
    for k in range(CRYSTAL.N):
        for j in range(k + 1, CRYSTAL.N):
            if CRYSTAL.topo.adjacency[k, j] > 0:
                r1, c1 = CRYSTAL.topo.node_positions[k]
                r2, c2 = CRYSTAL.topo.node_positions[j]
                ax.plot([c1, c2], [r1, r2], color='#334466', linewidth=0.3, alpha=0.4)
    
    ax.set_title("Crystal Reservoir State", color='#00c8ff', fontsize=10, fontfamily='monospace')
    ax.set_xlim(-2, CRYSTAL.topo.field_size + 2)
    ax.set_ylim(-2, CRYSTAL.topo.field_size + 2)
    ax.invert_yaxis()
    ax.axis('off')
    
    # 2. Eigenmode spectrum
    ax = axes[1]
    ax.set_facecolor('#0a0a14')
    mode_amps = viz['mode_amps']
    colors_mode = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD',
                   '#FF9FF3', '#48DBFB', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for m in range(len(mode_amps)):
        ax.bar(m, mode_amps[m], color=colors_mode[m % len(colors_mode)], 
               edgecolor='#444466', linewidth=0.5)
    
    ax.set_title("Eigenmode Spectrum", color='#34d399', fontsize=10, fontfamily='monospace')
    ax.set_xlabel("Mode", color='#667788', fontsize=8)
    ax.set_ylabel("Amplitude²", color='#667788', fontsize=8)
    ax.tick_params(colors='#556677', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    
    # 3. Band gains
    ax = axes[2]
    ax.set_facecolor('#0a0a14')
    band_names = ['STRUCT', 'MESO', 'MICRO', 'DETAIL']
    band_colors_hex = ['#00c8ff', '#ff6b35', '#a78bfa', '#34d399']
    
    # Get current gains (run a quick step)
    gains, _ = CRYSTAL.step(n_steps=1)
    gain_vals = [gains.get(b, 0) for b in band_names]
    
    bars = ax.barh(range(4), gain_vals, color=band_colors_hex, 
                   edgecolor='#444466', height=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels(band_names, color='#aabbcc', fontsize=9, fontfamily='monospace')
    ax.set_xlim(0, 1.1)
    ax.set_title("Spectral EQ Gains", color='#ff6b35', fontsize=10, fontfamily='monospace')
    ax.tick_params(colors='#556677', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333355')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#0a0a14', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)


def process_image(image, style, struct_base, meso_base, micro_base, detail_base,
                  crystal_depth, phase_gravity):
    """Main processing function for Gradio."""
    if image is None:
        return None, None, "Upload an image to begin."
    
    # Convert PIL → numpy
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Update blender settings
    BLENDER.base_gains = {
        'STRUCT': struct_base, 'MESO': meso_base,
        'MICRO': micro_base, 'DETAIL': detail_base,
    }
    BLENDER.crystal_depth = {
        'STRUCT': crystal_depth * 0.3,
        'MESO': crystal_depth * 0.5,
        'MICRO': crystal_depth * 0.8,
        'DETAIL': crystal_depth * 0.6,
    }
    BLENDER.phase_gravity = phase_gravity
    
    # Generate impulses from image
    impulses = image_to_impulses(img)
    CRYSTAL.inject_impulse(impulses)
    
    # Evolve crystal (more steps for single image)
    gains, mode_amps = CRYSTAL.step(n_steps=10)
    
    # Generate SD-like output
    DUMMY_SD.style = style.lower()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    sd_bgr = DUMMY_SD.generate(img_bgr)
    sd_rgb = cv2.cvtColor(sd_bgr, cv2.COLOR_BGR2RGB)
    
    # Blend
    output_bgr = BLENDER.blend(img_bgr, sd_bgr, gains)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    
    # Crystal visualization
    crystal_viz = plot_crystal_state()
    
    # Report
    report = f"**Crystal Response:**\n"
    report += f"- STRUCT: {gains['STRUCT']:.3f} | MESO: {gains['MESO']:.3f}\n"
    report += f"- MICRO: {gains['MICRO']:.3f} | DETAIL: {gains['DETAIL']:.3f}\n\n"
    report += f"**Top eigenmode energies:**\n"
    for m in range(min(6, len(mode_amps))):
        bar = "█" * int(mode_amps[m] / max(mode_amps.max(), 0.001) * 20)
        report += f"  λ_{m}: {mode_amps[m]:.4f} {bar}\n"
    
    return output_rgb, crystal_viz, report


# ═══ Gradio UI ═══

css = """
.gradio-container { max-width: 1100px !important; }
"""

with gr.Blocks(theme=gr.themes.Default(), css=css) as app:
    gr.Markdown("""
# 🔬 Resonant Cortex — Spectral Phase Lock
### *Optical Flow → Crystal Reservoir → Spectral EQ → Phase-Locked Blend*

Upload an image. The crystal's edge-response creates impulses that excite eigenmodes.
Different eigenmodes control different frequency bands of the output — 
like V1→V4 visual cortex processing, but grown from wave physics.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input (The Eye)", type="pil")
            
            style_dropdown = gr.Dropdown(
                choices=["Cyberpunk", "Dream"],
                value="Cyberpunk", label="Style (Memory Bank)"
            )
            
            gr.Markdown("### Geometric EQ (Base Gains)")
            struct_slider = gr.Slider(0, 1, 1.0, label="STRUCT (V1 — Global shape)", step=0.01)
            meso_slider = gr.Slider(0, 1, 1.0, label="MESO (V2 — Contour)", step=0.01)
            micro_slider = gr.Slider(0, 1, 0.0, label="MICRO (V4 — Form/identity)", step=0.01)
            detail_slider = gr.Slider(0, 1, 1.0, label="DETAIL (LOC — Texture)", step=0.01)
            
            gr.Markdown("### Crystal Dynamics")
            depth_slider = gr.Slider(0, 2, 1.0, label="Crystal Modulation Depth", step=0.05)
            gravity_slider = gr.Slider(0, 1, 0.0, label="Phase Gravity (drift to SD)", step=0.01)
            
            process_btn = gr.Button("⚡ Crystallize", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Output (Perception)")
            crystal_plot = gr.Image(label="Crystal Reservoir State")
            report_md = gr.Markdown(label="Crystal Report")
    
    process_btn.click(
        fn=process_image,
        inputs=[input_image, style_dropdown, 
                struct_slider, meso_slider, micro_slider, detail_slider,
                depth_slider, gravity_slider],
        outputs=[output_image, crystal_plot, report_md]
    )
    
    gr.Markdown("""
---
### How It Works

1. **Image gradients** (edges, contours) create complex impulses at crystal node positions
2. **Crystal reservoir** evolves under Clockfield dynamics: Γ = 1/(1 + τ·|z|²)²
3. **Eigenmode projection** decomposes the crystal response into spectral bands
4. **Phase-locked blend** mixes input (Eye) and SD output (Memory) per frequency band
5. High crystal activity → grounded in reality | Low activity → hallucinating from memory

*The crystal IS the cortex. The webcam IS the eye. SD IS memory. This IS perception.*

PerceptionLab / Antti Luode — Deerskin Architecture
    """)


if __name__ == "__main__":
    app.launch()
