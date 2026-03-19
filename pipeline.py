"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RESONANT CORTEX — Main Pipeline                                             ║
║                                                                              ║
║  The full loop:                                                              ║
║    1. Read webcam frame                                                      ║
║    2. Compute optical flow (retinal motion)                                  ║
║    3. Convert flow to complex impulses at crystal nodes                      ║
║    4. Inject impulses, evolve crystal reservoir                              ║
║    5. Read eigenmode amplitudes → spectral EQ gains                          ║
║    6. Decompose webcam + SD into frequency bands                             ║
║    7. Blend per-band using crystal gains                                     ║
║    8. Display                                                                ║
║                                                                              ║
║  Run: python pipeline.py                                                     ║
║  Run without SD: python pipeline.py --no-sd                                  ║
║  Run with SD: python pipeline.py --sd-model runwayml/stable-diffusion-v1-5   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import cv2
import time
import argparse
from typing import Optional

from crystal_reservoir import grow_crystal, CrystalReservoir
from optical_flow import OpticalFlowExtractor, WebcamCapture
from spectral_eq import SpectralEQ, PhaseLockedBlender
from sd_interface import create_sd_interface


class ResonantCortexPipeline:
    """
    Main pipeline connecting all components.
    """
    
    def __init__(self, 
                 crystal_size: int = 48,
                 crystal_beta: float = 0.8,
                 crystal_steps: int = 300,
                 crystal_modes: int = 12,
                 crystal_seed: int = 42,
                 sd_model: Optional[str] = None,
                 sd_device: str = "cuda",
                 sd_strength: float = 0.33,
                 sd_prompt: str = "cyberpunk portrait, neon lighting",
                 cam_device: int = 0,
                 cam_width: int = 640,
                 cam_height: int = 480,
                 sd_interval: int = 10):
        """
        Args:
            sd_interval: Run SD every N frames (SD is slow, don't block every frame)
        """
        print("=" * 60)
        print("  RESONANT CORTEX — Dynamic Reservoir Bridge")
        print("  Optical Flow → Crystal → Spectral EQ → Phase-Locked Blend")
        print("=" * 60)
        
        # Grow crystal
        print("\n[1/4] Growing crystal reservoir...")
        self.crystal = grow_crystal(
            size=crystal_size, beta=crystal_beta,
            steps=crystal_steps, seed=crystal_seed,
            n_modes=crystal_modes
        )
        
        # Optical flow
        print("[2/4] Initializing optical flow extractor...")
        self.flow_ext = OpticalFlowExtractor(downscale=4, flow_scale=0.15)
        
        # Spectral EQ + Blender
        print("[3/4] Setting up spectral EQ...")
        self.blender = PhaseLockedBlender()
        self.blender.sd_strength = sd_strength
        
        # Stable Diffusion
        print("[4/4] Setting up image generator...")
        self.sd_prompt = sd_prompt
        self.sd_interval = sd_interval
        self.use_sd = sd_model is not None
        
        if self.use_sd:
            self.sd = create_sd_interface(model_id=sd_model, device=sd_device)
            self.sd.set_strength(sd_strength)
        else:
            self.sd = None
        
        # State
        self.last_sd_frame = None
        self.frame_count = 0
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        # Precompute node screen positions
        self.node_screen_pos = self.crystal.get_node_screen_positions(cam_width, cam_height)
        
        print(f"\nReady. Crystal: {self.crystal.N} nodes, "
              f"{self.crystal.M} eigenmodes, SD: {'ON' if self.use_sd else 'OFF'}")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single webcam frame through the full pipeline.
        
        Returns dict with:
            'output': blended output frame
            'crystal_gains': current spectral EQ gains
            'mode_amps': eigenmode amplitudes
            'flow_mag': optical flow magnitude (for visualization)
        """
        self.frame_count += 1
        
        # 1. Compute optical flow
        flow = self.flow_ext.compute_flow(frame)
        flow_mag = None
        
        if flow is not None:
            # 2. Convert to crystal impulses
            impulses = self.flow_ext.flow_to_impulses(
                flow, self.node_screen_pos,
                self.cam_width, self.cam_height
            )
            
            # 3. Inject into crystal
            self.crystal.inject_impulse(impulses)
            
            # Flow magnitude for visualization
            flow_mag = self.flow_ext.get_flow_magnitude_map(flow)
        
        # 4. Evolve crystal and read gains
        crystal_gains, mode_amps = self.crystal.step(n_steps=3)
        
        # 5. Generate SD frame (periodically)
        if self.use_sd and self.sd is not None:
            if self.last_sd_frame is None or self.frame_count % self.sd_interval == 0:
                sd_out = self.sd.generate(frame, prompt=self.sd_prompt)
                if sd_out is not None:
                    self.last_sd_frame = sd_out
        
        # 6. Blend
        if self.last_sd_frame is not None:
            output = self.blender.blend(frame, self.last_sd_frame, crystal_gains)
        else:
            output = self.blender.blend_without_sd(frame, crystal_gains)
        
        return {
            'output': output,
            'crystal_gains': crystal_gains,
            'mode_amps': mode_amps,
            'flow_mag': flow_mag,
        }
    
    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw crystal and EQ overlay on the output frame."""
        output = result['output'].copy()
        gains = result['crystal_gains']
        mode_amps = result['mode_amps']
        
        h, w = output.shape[:2]
        
        # === EQ Bars (bottom-left) ===
        bar_x, bar_y = 20, h - 120
        bar_w, bar_h_max = 50, 80
        band_colors = {
            'STRUCT': (255, 200, 0),    # Cyan-ish (BGR)
            'MESO':   (0, 200, 200),    # Yellow
            'MICRO':  (200, 100, 255),  # Pink
            'DETAIL': (0, 255, 100),    # Green
        }
        
        for i, band_name in enumerate(SpectralEQ.BAND_NAMES):
            x = bar_x + i * (bar_w + 10)
            gain = gains.get(band_name, 0.5)
            bar_h = int(gain * bar_h_max)
            
            color = band_colors[band_name]
            
            # Background
            cv2.rectangle(output, (x, bar_y), (x + bar_w, bar_y + bar_h_max),
                         (30, 30, 40), -1)
            # Fill
            cv2.rectangle(output, (x, bar_y + bar_h_max - bar_h), 
                         (x + bar_w, bar_y + bar_h_max), color, -1)
            # Label
            cv2.putText(output, band_name[:4], (x + 2, bar_y + bar_h_max + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            # Value
            cv2.putText(output, f"{gain:.2f}", (x + 2, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 200), 1)
        
        # === Crystal Nodes (top-right) ===
        viz = self.crystal.get_state_visualization()
        node_mags = viz['node_magnitudes']
        max_mag = max(node_mags.max(), 0.001)
        
        # Draw in a small box
        box_x, box_y = w - 180, 20
        box_size = 150
        
        cv2.rectangle(output, (box_x, box_y), (box_x + box_size, box_y + box_size),
                     (20, 20, 30), -1)
        cv2.putText(output, "CRYSTAL", (box_x + 5, box_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 255), 1)
        
        for k in range(self.crystal.N):
            row, col = self.crystal.topo.node_positions[k]
            nx = int(box_x + col / self.crystal.topo.field_size * box_size)
            ny = int(box_y + row / self.crystal.topo.field_size * box_size)
            
            mag_norm = node_mags[k] / max_mag
            radius = max(1, int(2 + mag_norm * 4))
            brightness = int(50 + mag_norm * 205)
            
            cv2.circle(output, (nx, ny), radius, (brightness, brightness // 2, 0), -1)
        
        # === Eigenmode Spectrum (top-left) ===
        spec_x, spec_y = 20, 20
        spec_w = 200
        max_amp = max(mode_amps.max(), 0.001)
        
        cv2.putText(output, "EIGENMODES", (spec_x, spec_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 150), 1)
        
        mode_colors = [
            (70, 70, 255), (75, 200, 78), (210, 130, 55),
            (200, 80, 210), (50, 180, 250), (120, 220, 220),
        ]
        
        for m in range(min(len(mode_amps), 12)):
            bar_h_m = int(mode_amps[m] / max_amp * 30)
            bx = spec_x + m * 16
            by = spec_y + 45
            
            color = mode_colors[m % len(mode_colors)]
            cv2.rectangle(output, (bx, by - bar_h_m), (bx + 12, by), color, -1)
        
        # === FPS ===
        cv2.putText(output, f"Frame {self.frame_count}", (w - 120, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 120), 1)
        
        return output


def main():
    parser = argparse.ArgumentParser(description="Resonant Cortex Pipeline")
    parser.add_argument('--no-sd', action='store_true', help='Disable Stable Diffusion')
    parser.add_argument('--sd-model', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--sd-device', type=str, default='cuda')
    parser.add_argument('--sd-strength', type=float, default=0.33)
    parser.add_argument('--prompt', type=str, default='cyberpunk portrait, neon lighting')
    parser.add_argument('--cam', type=int, default=0, help='Camera device index')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--crystal-size', type=int, default=48)
    parser.add_argument('--crystal-beta', type=float, default=0.8)
    parser.add_argument('--crystal-modes', type=int, default=12)
    parser.add_argument('--crystal-seed', type=int, default=42)
    args = parser.parse_args()
    
    sd_model = None if args.no_sd else args.sd_model
    
    pipeline = ResonantCortexPipeline(
        crystal_size=args.crystal_size,
        crystal_beta=args.crystal_beta,
        crystal_modes=args.crystal_modes,
        crystal_seed=args.crystal_seed,
        sd_model=sd_model,
        sd_device=args.sd_device,
        sd_strength=args.sd_strength,
        sd_prompt=args.prompt,
        cam_device=args.cam,
        cam_width=args.width,
        cam_height=args.height,
    )
    
    # Open webcam
    cam = WebcamCapture(device=args.cam, width=args.width, height=args.height)
    
    print("\n" + "=" * 60)
    print("  LIVE — Press 'q' to quit, 's' to toggle SD, +/- for strength")
    print("=" * 60)
    
    while True:
        frame = cam.read()
        if frame is None:
            print("Camera read failed.")
            break
        
        t0 = time.time()
        result = pipeline.process_frame(frame)
        output = pipeline.draw_overlay(frame, result)
        dt = time.time() - t0
        
        # Show FPS
        fps = 1.0 / max(dt, 0.001)
        cv2.putText(output, f"FPS: {fps:.1f}", (output.shape[1] - 100, output.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 180), 1)
        
        cv2.imshow("Resonant Cortex", output)
        
        # Also show flow if available
        if result['flow_mag'] is not None:
            flow_display = cv2.resize(result['flow_mag'], (320, 240))
            flow_color = cv2.applyColorMap(flow_display, cv2.COLORMAP_INFERNO)
            cv2.imshow("Optical Flow (Retinal Motion)", flow_color)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            pipeline.use_sd = not pipeline.use_sd
            print(f"  SD: {'ON' if pipeline.use_sd else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            pipeline.blender.sd_strength = min(1.0, pipeline.blender.sd_strength + 0.05)
            print(f"  SD strength: {pipeline.blender.sd_strength:.2f}")
        elif key == ord('-'):
            pipeline.blender.sd_strength = max(0.0, pipeline.blender.sd_strength - 0.05)
            print(f"  SD strength: {pipeline.blender.sd_strength:.2f}")
        elif key == ord('g'):
            pipeline.blender.phase_gravity = min(1.0, pipeline.blender.phase_gravity + 0.05)
            print(f"  Phase gravity: {pipeline.blender.phase_gravity:.2f}")
        elif key == ord('h'):
            pipeline.blender.phase_gravity = max(0.0, pipeline.blender.phase_gravity - 0.05)
            print(f"  Phase gravity: {pipeline.blender.phase_gravity:.2f}")
    
    cam.release()
    cv2.destroyAllWindows()
    print("\nResonant Cortex shutdown.")


if __name__ == "__main__":
    main()
