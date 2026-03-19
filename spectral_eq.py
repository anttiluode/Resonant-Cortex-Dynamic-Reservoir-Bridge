"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SPECTRAL EQ — Four-Band Frequency Decomposition of Images                  ║
║                                                                              ║
║  Implements the V1→V4 visual cortex hierarchy as a Laplacian pyramid.        ║
║                                                                              ║
║  Band     Neural Analog    Spatial Frequency                                 ║
║  STRUCT   V1 (Gist)        Lowest — global silhouette                        ║
║  MESO     V2 (Contour)     Low-mid — spatial arrangement                     ║
║  MICRO    V4 (Form)        High-mid — complex shape, identity                ║
║  DETAIL   LOC (Texture)    Highest — surface texture, fine style             ║
║                                                                              ║
║  Each band is independently blended between webcam and SD output             ║
║  using gains from the crystal reservoir.                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import cv2
from typing import Dict, Tuple


class SpectralEQ:
    """
    Four-band image frequency decomposition using Laplacian pyramid.
    
    Decomposes an image into:
      STRUCT  — 1/8 resolution base (global structure)
      MESO    — 1/4 → 1/8 residual (contour)
      MICRO   — 1/2 → 1/4 residual (form)
      DETAIL  — full → 1/2 residual (texture)
    
    Then reconstructs from bands with per-band gain control.
    """
    
    BAND_NAMES = ['STRUCT', 'MESO', 'MICRO', 'DETAIL']
    
    def __init__(self, levels: int = 4):
        """
        Args:
            levels: Number of pyramid levels (4 = four bands)
        """
        self.levels = levels
    
    def decompose(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decompose image into frequency bands.
        
        Args:
            image: (H, W, 3) uint8 or float32 image
        
        Returns:
            bands: dict mapping band name → image array
                   All bands are stored at the ORIGINAL resolution
                   for easy blending.
        """
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:
            img = image.astype(np.float32)
        
        # Build Gaussian pyramid
        gaussian_pyramid = [img]
        for _ in range(self.levels - 1):
            blurred = cv2.GaussianBlur(gaussian_pyramid[-1], (5, 5), 0)
            down = cv2.resize(blurred, None, fx=0.5, fy=0.5, 
                            interpolation=cv2.INTER_LINEAR)
            gaussian_pyramid.append(down)
        
        # Build Laplacian pyramid (residuals at each level)
        laplacian_pyramid = []
        for i in range(self.levels - 1):
            h, w = gaussian_pyramid[i].shape[:2]
            upsampled = cv2.resize(gaussian_pyramid[i + 1], (w, h),
                                   interpolation=cv2.INTER_LINEAR)
            residual = gaussian_pyramid[i] - upsampled
            laplacian_pyramid.append(residual)
        
        # Base (lowest frequency)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        
        # Map to band names (reverse order: base = STRUCT, finest = DETAIL)
        # laplacian_pyramid[-1] = base = STRUCT
        # laplacian_pyramid[0] = finest residual = DETAIL
        H, W = img.shape[:2]
        bands = {}
        band_names_reversed = list(reversed(self.BAND_NAMES))
        
        for i, name in enumerate(band_names_reversed):
            if i < len(laplacian_pyramid):
                band = laplacian_pyramid[len(laplacian_pyramid) - 1 - i]
                # Upsample to original resolution for blending
                if band.shape[0] != H or band.shape[1] != W:
                    band = cv2.resize(band, (W, H), interpolation=cv2.INTER_LINEAR)
                bands[name] = band
            else:
                bands[name] = np.zeros_like(img)
        
        return bands
    
    def reconstruct(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Reconstruct image from frequency bands.
        
        Args:
            bands: dict mapping band name → image array (all same resolution)
        
        Returns:
            image: (H, W, 3) float32 image
        """
        result = np.zeros_like(bands['STRUCT'], dtype=np.float32)
        for name in self.BAND_NAMES:
            if name in bands:
                result += bands[name]
        return np.clip(result, 0, 1)


class PhaseLockedBlender:
    """
    Blends webcam and SD images per-frequency-band using crystal-driven gains.
    
    For each band:
        output_band = gain · webcam_band + (1 - gain) · sd_band
    
    gain = 1.0 → fully grounded in reality (webcam)
    gain = 0.0 → fully hallucinating from memory (SD)
    
    The crystal reservoir controls these gains dynamically based on 
    the optical flow from the webcam.
    """
    
    def __init__(self):
        self.eq = SpectralEQ(levels=4)
        
        # Default gains (user can override)
        self.base_gains = {
            'STRUCT': 1.0,   # Structure from webcam by default
            'MESO':   1.0,   # Contour from webcam
            'MICRO':  0.0,   # Form from SD
            'DETAIL': 1.0,   # Texture from webcam
        }
        
        # Crystal modulation depth: how much the crystal can move each band
        self.crystal_depth = {
            'STRUCT': 0.3,
            'MESO':   0.5,
            'MICRO':  0.8,
            'DETAIL': 0.6,
        }
        
        # Phase gravity: how quickly bands drift toward SD when no motion
        self.phase_gravity = 0.0  # 0 = no drift, 1 = instant drift to SD
        
        # SD diffusion strength (0-1)
        self.sd_strength = 0.33
    
    def blend(self, webcam_frame: np.ndarray,
              sd_frame: np.ndarray,
              crystal_gains: Dict[str, float]) -> np.ndarray:
        """
        Blend webcam and SD frames using spectral phase-lock.
        
        Args:
            webcam_frame: (H, W, 3) webcam image
            sd_frame: (H, W, 3) stable diffusion output (same size)
            crystal_gains: {STRUCT, MESO, MICRO, DETAIL} → [0, 1]
                          from crystal reservoir
        
        Returns:
            blended: (H, W, 3) output image
        """
        # Ensure same size
        h, w = webcam_frame.shape[:2]
        if sd_frame.shape[0] != h or sd_frame.shape[1] != w:
            sd_frame = cv2.resize(sd_frame, (w, h))
        
        # Decompose both into frequency bands
        cam_bands = self.eq.decompose(webcam_frame)
        sd_bands = self.eq.decompose(sd_frame)
        
        # Blend per band
        output_bands = {}
        for band_name in SpectralEQ.BAND_NAMES:
            # Compute effective gain
            base = self.base_gains[band_name]
            crystal_mod = crystal_gains.get(band_name, 0.5)
            depth = self.crystal_depth[band_name]
            
            # Crystal modulates around base: when crystal is excited,
            # it pushes the gain TOWARD webcam (grounding in reality)
            # When crystal is quiet, phase gravity pulls toward SD
            effective_gain = base + depth * (crystal_mod - 0.5) * 2
            
            # Phase gravity: drift toward SD when quiet
            if self.phase_gravity > 0:
                effective_gain -= self.phase_gravity * (1 - crystal_mod) * depth
            
            effective_gain = np.clip(effective_gain, 0, 1)
            
            # Blend
            cam = cam_bands[band_name]
            sd = sd_bands[band_name]
            output_bands[band_name] = effective_gain * cam + (1 - effective_gain) * sd
        
        # Reconstruct
        result = self.eq.reconstruct(output_bands)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def blend_without_sd(self, webcam_frame: np.ndarray,
                         crystal_gains: Dict[str, float]) -> np.ndarray:
        """
        When no SD is available, modulate the webcam itself:
        suppress quiet bands, boost active bands → 'crystal vision' effect.
        """
        cam_bands = self.eq.decompose(webcam_frame)
        
        output_bands = {}
        for band_name in SpectralEQ.BAND_NAMES:
            crystal_mod = crystal_gains.get(band_name, 0.5)
            depth = self.crystal_depth[band_name]
            
            # Amplify bands where crystal is excited, suppress where quiet
            gain = 0.5 + depth * (crystal_mod - 0.5) * 2
            gain = np.clip(gain, 0.1, 2.0)
            
            output_bands[band_name] = cam_bands[band_name] * gain
        
        result = self.eq.reconstruct(output_bands)
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    # Test decomposition and reconstruction
    print("SpectralEQ test: decompose → reconstruct identity check")
    
    eq = SpectralEQ()
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    bands = eq.decompose(test_img)
    print(f"  Bands: {list(bands.keys())}")
    for name, band in bands.items():
        print(f"    {name}: shape={band.shape}, range=[{band.min():.3f}, {band.max():.3f}]")
    
    reconstructed = eq.reconstruct(bands)
    original = test_img.astype(np.float32) / 255.0
    error = np.mean(np.abs(reconstructed - original))
    print(f"  Reconstruction error: {error:.6f}")
    print(f"  {'✓ PASS' if error < 0.01 else '✗ FAIL'}")
