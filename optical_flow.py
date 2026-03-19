"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  OPTICAL FLOW — The Retina                                                   ║
║                                                                              ║
║  Extracts motion vectors from consecutive webcam frames and converts         ║
║  them into complex impulses that strike crystal nodes.                        ║
║                                                                              ║
║  Flow → Complex impulse: magnitude · exp(i · direction)                      ║
║                                                                              ║
║  Fast motion → large impulse → excites high-frequency eigenmodes             ║
║  Slow motion → small impulse → excites low-frequency eigenmodes              ║
║  This IS the V1→V4 hierarchy emerging from physics.                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class OpticalFlowExtractor:
    """
    Extracts dense optical flow from webcam and maps it to crystal node impulses.
    
    Uses Farneback dense optical flow (GPU not required).
    """
    
    def __init__(self, downscale: int = 4,
                 flow_scale: float = 0.1,
                 blur_sigma: float = 3.0):
        """
        Args:
            downscale: Factor to shrink frames before flow computation (speed vs accuracy)
            flow_scale: Multiplier converting pixel flow to impulse magnitude
            blur_sigma: Gaussian blur on flow field before sampling at nodes
        """
        self.downscale = downscale
        self.flow_scale = flow_scale
        self.blur_sigma = blur_sigma
        self.prev_gray = None
    
    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute dense optical flow from current frame.
        
        Args:
            frame: BGR frame from cv2.VideoCapture, shape (H, W, 3)
        
        Returns:
            flow: (H_small, W_small, 2) array of (dx, dy) vectors,
                  or None if this is the first frame.
        """
        # Convert and downscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small_h, small_w = h // self.downscale, w // self.downscale
        gray_small = cv2.resize(gray, (small_w, small_h))
        
        if self.prev_gray is None:
            self.prev_gray = gray_small
            return None
        
        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray_small,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        self.prev_gray = gray_small
        return flow
    
    def flow_to_impulses(self, flow: np.ndarray,
                         node_screen_positions: np.ndarray,
                         frame_width: int,
                         frame_height: int) -> np.ndarray:
        """
        Sample the optical flow field at crystal node positions and convert
        to complex impulses.
        
        Args:
            flow: (H_small, W_small, 2) dense flow field
            node_screen_positions: (N, 2) array of (x, y) in pixel coords
            frame_width: original frame width
            frame_height: original frame height
        
        Returns:
            impulses: (N,) complex array — each node's impulse from motion
        """
        N = len(node_screen_positions)
        flow_h, flow_w = flow.shape[:2]
        
        # Smooth flow field to reduce noise
        if self.blur_sigma > 0:
            flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (0, 0), self.blur_sigma)
            flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (0, 0), self.blur_sigma)
        
        impulses = np.zeros(N, dtype=np.complex128)
        
        for k in range(N):
            # Map node screen position to flow field coordinates
            sx, sy = node_screen_positions[k]
            fx = int(np.clip(sx / frame_width * flow_w, 0, flow_w - 1))
            fy = int(np.clip(sy / frame_height * flow_h, 0, flow_h - 1))
            
            dx = flow[fy, fx, 0]
            dy = flow[fy, fx, 1]
            
            magnitude = np.sqrt(dx * dx + dy * dy) * self.flow_scale
            angle = np.arctan2(dy, dx)
            
            # Complex impulse: magnitude * exp(i * angle)
            impulses[k] = magnitude * np.exp(1j * angle)
        
        return impulses
    
    def get_flow_magnitude_map(self, flow: np.ndarray) -> np.ndarray:
        """Return a visualization-friendly magnitude map from flow."""
        mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
        # Normalize to [0, 255]
        mag = np.clip(mag * 10, 0, 255).astype(np.uint8)
        return mag


class WebcamCapture:
    """Simple webcam interface."""
    
    def __init__(self, device: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame. Returns BGR array or None."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Mirror for face-cam feel
        frame = cv2.flip(frame, 1)
        return frame
    
    def release(self):
        self.cap.release()
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.width, self.height)


if __name__ == "__main__":
    # Quick test: show optical flow from webcam
    cam = WebcamCapture()
    flow_ext = OpticalFlowExtractor()
    
    print("Webcam optical flow test. Press 'q' to quit.")
    
    while True:
        frame = cam.read()
        if frame is None:
            break
        
        flow = flow_ext.compute_flow(frame)
        
        if flow is not None:
            mag_map = flow_ext.get_flow_magnitude_map(flow)
            # Upscale for display
            mag_display = cv2.resize(mag_map, (cam.width, cam.height))
            mag_color = cv2.applyColorMap(mag_display, cv2.COLORMAP_INFERNO)
            
            # Blend with original
            combined = cv2.addWeighted(frame, 0.6, mag_color, 0.4, 0)
            cv2.imshow("Optical Flow + Webcam", combined)
        else:
            cv2.imshow("Optical Flow + Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
