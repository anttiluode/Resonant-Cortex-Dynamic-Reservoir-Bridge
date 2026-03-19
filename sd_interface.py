"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SD INTERFACE — Stable Diffusion as "Memory"                                 ║
║                                                                              ║
║  Wraps diffusers img2img pipeline.                                           ║
║  The SD output is "what the brain thinks it's seeing" — the top-down         ║
║  prediction from memory, before being corrected by bottom-up sensory         ║
║  input (the webcam).                                                         ║
║                                                                              ║
║  In the Deerskin framework:                                                  ║
║    Webcam  = bottom-up sensory input (the Eye)                               ║
║    SD      = top-down memory prediction (the Hippocampus)                    ║
║    Crystal = the cortical resonance that decides how much of each to trust   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from PIL import Image
from typing import Optional


class SDInterface:
    """
    Stable Diffusion img2img interface.
    
    Lazy-loads the pipeline only when first called.
    Can run without SD installed (returns None).
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda", strength: float = 0.33,
                 guidance_scale: float = 7.5, num_steps: int = 15):
        self.model_id = model_id
        self.device = device
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self.pipe = None
        self._available = None
    
    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import torch
                from diffusers import StableDiffusionImg2ImgPipeline
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def _load(self):
        """Lazy-load the pipeline."""
        if self.pipe is not None:
            return
        
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        
        print(f"Loading Stable Diffusion ({self.model_id})...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
        
        print("  SD pipeline ready.")
    
    def generate(self, webcam_frame: np.ndarray,
                 prompt: str = "cyberpunk portrait, neon lighting",
                 negative_prompt: str = "blurry, low quality") -> Optional[np.ndarray]:
        """
        Run img2img on a webcam frame.
        
        Args:
            webcam_frame: (H, W, 3) BGR uint8 frame from OpenCV
            prompt: text prompt for SD
            negative_prompt: negative prompt
        
        Returns:
            sd_output: (H, W, 3) BGR uint8 image, or None if SD unavailable
        """
        if not self.available:
            return None
        
        self._load()
        
        # Convert BGR→RGB and to PIL
        rgb = webcam_frame[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        
        # Resize to SD-friendly dimensions
        w, h = pil_img.size
        # Round to nearest 64
        new_w = max(64, (w // 64) * 64)
        new_h = max(64, (h // 64) * 64)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        
        # Run img2img
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_img,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_steps,
        )
        
        sd_img = result.images[0]
        
        # Resize back to original and convert to BGR
        sd_img = sd_img.resize((w, h), Image.LANCZOS)
        sd_arr = np.array(sd_img)[:, :, ::-1]  # RGB→BGR
        
        return sd_arr
    
    def set_strength(self, strength: float):
        """Set denoising strength (0 = identical to input, 1 = pure generation)."""
        self.strength = np.clip(strength, 0.0, 1.0)


class DummySD:
    """
    Fallback when SD is not available.
    
    Produces a stylized version of the webcam frame using pure OpenCV:
    color shift + posterize + edge glow. Not great but gives something 
    to blend with.
    """
    
    def __init__(self, style: str = "cyberpunk"):
        self.style = style
    
    @property
    def available(self) -> bool:
        return True
    
    def generate(self, webcam_frame: np.ndarray, 
                 prompt: str = "", **kwargs) -> np.ndarray:
        """Generate a stylized version of the webcam frame."""
        import cv2
        
        frame = webcam_frame.copy()
        
        if self.style == "cyberpunk":
            # Color shift toward cyan/magenta
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180  # hue shift
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)  # saturation boost
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            # Posterize
            frame = (frame // 48) * 48
            
            # Edge glow
            edges = cv2.Canny(webcam_frame, 50, 150)
            edges_color = cv2.applyColorMap(edges, cv2.COLORMAP_OCEAN)
            frame = cv2.addWeighted(frame, 0.7, edges_color, 0.3, 0)
        
        elif self.style == "dream":
            # Soft blur + high saturation
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return frame
    
    def set_strength(self, strength: float):
        pass


def create_sd_interface(model_id: str = "runwayml/stable-diffusion-v1-5",
                         device: str = "cuda",
                         fallback_style: str = "cyberpunk"):
    """
    Create SD interface, falling back to DummySD if diffusers not available.
    """
    sd = SDInterface(model_id=model_id, device=device)
    if sd.available:
        return sd
    else:
        print("  Stable Diffusion not available (install torch + diffusers).")
        print(f"  Using DummySD fallback (style: {fallback_style})")
        return DummySD(style=fallback_style)
