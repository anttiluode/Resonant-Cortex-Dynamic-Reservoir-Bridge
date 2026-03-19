"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CRYSTAL RESERVOIR — Physics-Grown Dynamical System                          ║
║                                                                              ║
║  A Moiré crystal grown from the nonlinear Schrödinger equation,              ║
║  whose nodes hold complex state that evolves under:                          ║
║    - Graph Laplacian diffusion (coupling)                                    ║
║    - Clockfield self-trapping Γ = 1/(1 + τ·|z|²)²                          ║
║    - External impulses from optical flow                                     ║
║                                                                              ║
║  The eigenmode projections of this evolving state become the                 ║
║  spectral EQ gains that control the webcam↔SD blend.                        ║
║                                                                              ║
║  Antti Luode / PerceptionLab — Architecture: Claude (Opus 4.6)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import KDTree
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrystalTopology:
    """Frozen topology extracted from a PhiField."""
    node_positions: np.ndarray      # (N, 2) — positions in [0, field_size)
    node_amplitudes: np.ndarray     # (N,)
    node_phases: np.ndarray         # (N,)
    adjacency: np.ndarray           # (N, N) — weighted adjacency matrix
    eigenvalues: np.ndarray         # (M,) — Laplacian eigenvalues (skip λ₀=0)
    eigenvectors: np.ndarray        # (N, M) — corresponding eigenvectors
    field_size: int


class PhiFieldGrower:
    """
    Grow a 2D complex NLS field and extract crystal topology.
    
    Physics:
        dψ/dt = i·D·∇²ψ − i·β·|ψ|²·ψ + noise
        Γ = 1/(1 + 0.5·|∇²ψ|)²   (Clockfield time dilation)
    """
    
    def __init__(self, size: int = 48, beta: float = 0.8, diffusion: float = 1.0,
                 seed: int = 42, dt: float = 0.02, noise_scale: float = 0.005,
                 clamp_max: float = 3.0):
        self.size = size
        self.beta = beta
        self.diffusion = diffusion
        self.dt = dt
        self.noise_scale = noise_scale
        self.clamp_max = clamp_max
        self.rng = np.random.default_rng(seed)
        
        # Initialize complex field
        real = self.rng.standard_normal((size, size))
        imag = self.rng.standard_normal((size, size))
        self.psi = (real + 1j * imag) / np.sqrt(2.0)
    
    def evolve(self, steps: int = 300):
        """Run the NLS field forward."""
        for _ in range(steps):
            # Laplacian (periodic boundary)
            lap = (np.roll(self.psi, 1, 0) + np.roll(self.psi, -1, 0) +
                   np.roll(self.psi, 1, 1) + np.roll(self.psi, -1, 1) - 4 * self.psi)
            
            # Dispersive kinetic + defocusing nonlinearity
            kinetic = 1j * self.diffusion * lap
            amp_sq = np.abs(self.psi) ** 2
            nonlinear = -1j * self.beta * amp_sq * self.psi
            
            # Clockfield metric
            gamma = 1.0 / (1.0 + 0.5 * np.abs(lap)) ** 2
            
            # Noise
            noise = self.noise_scale * (
                self.rng.standard_normal(self.psi.shape) +
                1j * self.rng.standard_normal(self.psi.shape)
            )
            
            # Update
            self.psi += self.dt * (kinetic + nonlinear + noise) * gamma
            
            # Local soft-clamp
            amp = np.abs(self.psi)
            mask = amp > self.clamp_max
            if np.any(mask):
                scale = np.ones_like(amp)
                scale[mask] = self.clamp_max / amp[mask]
                self.psi *= scale
    
    def extract_topology(self, amp_thresh_sigma: float = 0.5,
                         coherence_thresh: float = 0.05,
                         conn_radius: float = 15.0,
                         min_peak_dist: int = 3,
                         n_modes: int = 12) -> CrystalTopology:
        """
        Freeze the field and extract crystal topology with eigenmodes.
        
        n_modes: how many Laplacian eigenmodes to compute (skip mode 0).
                 More modes = finer spectral resolution for the EQ.
        """
        amp = np.abs(self.psi)
        phase = np.angle(self.psi)
        
        # Smooth and find peaks
        smooth_amp = gaussian_filter(amp, sigma=1.5)
        footprint = min_peak_dist * 2 + 1
        local_max = maximum_filter(smooth_amp, size=footprint) == smooth_amp
        threshold = np.mean(smooth_amp) + amp_thresh_sigma * np.std(smooth_amp)
        peak_mask = local_max & (smooth_amp > threshold)
        peak_coords = np.argwhere(peak_mask)
        
        if len(peak_coords) < 4:
            # Fallback: top-k
            k = min(30, amp.size)
            flat_idx = np.argsort(smooth_amp.ravel())[-k:]
            peak_coords = np.array(np.unravel_index(flat_idx, smooth_amp.shape)).T
        
        # Limit nodes
        peak_coords = peak_coords[:50]
        N = len(peak_coords)
        
        node_amps = np.array([amp[r, c] for r, c in peak_coords])
        node_phases = np.array([phase[r, c] for r, c in peak_coords])
        
        # Build adjacency via Moiré score
        tree = KDTree(peak_coords)
        pairs = tree.query_pairs(r=conn_radius)
        
        adj = np.zeros((N, N), dtype=np.float64)
        for i, j in pairs:
            phi_i = self.psi[peak_coords[i][0], peak_coords[i][1]]
            phi_j = self.psi[peak_coords[j][0], peak_coords[j][1]]
            score = float(np.real(phi_i * np.conj(phi_j)))
            if abs(score) >= coherence_thresh:
                adj[i, j] = abs(score)
                adj[j, i] = abs(score)
        
        # Compute symmetric normalized Laplacian eigenmodes
        degrees = adj.sum(axis=1)
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 0
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt
        L = (L + L.T) / 2.0  # numerical symmetry
        
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # Skip mode 0 (constant), take n_modes
        actual_modes = min(n_modes, N - 1)
        eigenvalues = eigenvalues[1:actual_modes + 1]
        eigenvectors = eigenvectors[:, 1:actual_modes + 1]
        
        return CrystalTopology(
            node_positions=peak_coords.astype(np.float64),
            node_amplitudes=node_amps,
            node_phases=node_phases,
            adjacency=adj,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            field_size=self.size,
        )


class CrystalReservoir:
    """
    Dynamical reservoir running on a frozen crystal topology.
    
    Each node holds a complex state z_k(t) that evolves under:
      - Graph diffusion: coupling * Σ_j A_{kj} · (z_j - z_k)
      - Self-trapping:  -beta * |z_k|² * z_k
      - Clockfield:     Γ_k = 1/(1 + tau * |z_k|²)²
      - Damping:        -damping * z_k
      - External input: impulse_k(t)
    
    The eigenmode projections a_m(t) = Σ_k z_k · v_m(k) are the output.
    """
    
    def __init__(self, topology: CrystalTopology,
                 coupling: float = 0.5,
                 self_trap_beta: float = 0.3,
                 tau: float = 1.0,
                 damping: float = 0.05,
                 dt: float = 0.02,
                 ema_alpha: float = 0.15):
        self.topo = topology
        self.N = len(topology.node_positions)
        self.M = len(topology.eigenvalues)
        self.coupling = coupling
        self.beta = self_trap_beta
        self.tau = tau
        self.damping = damping
        self.dt = dt
        self.ema_alpha = ema_alpha
        
        # Node states
        self.z = np.zeros(self.N, dtype=np.complex128)
        
        # Eigenmode amplitudes (smoothed)
        self.mode_amps = np.zeros(self.M, dtype=np.float64)
        
        # Precompute normalized adjacency for diffusion
        # L_diff[k] = list of (j, weight) for neighbors of k
        self.neighbors = []
        for k in range(self.N):
            nbrs = []
            for j in range(self.N):
                if self.topo.adjacency[k, j] > 0 and k != j:
                    nbrs.append((j, self.topo.adjacency[k, j]))
            self.neighbors.append(nbrs)
        
        # EQ band assignments: which eigenmodes map to which band
        # STRUCT=low modes, MESO=mid-low, MICRO=mid-high, DETAIL=high
        self.band_ranges = self._assign_bands()
    
    def _assign_bands(self):
        """Divide eigenmodes into 4 spectral bands."""
        M = self.M
        if M < 4:
            # Too few modes — spread evenly
            return {
                'STRUCT': list(range(min(1, M))),
                'MESO':   list(range(min(1, M), min(2, M))),
                'MICRO':  list(range(min(2, M), min(3, M))),
                'DETAIL': list(range(min(3, M), M)),
            }
        
        q1 = M // 4
        q2 = M // 2
        q3 = (3 * M) // 4
        return {
            'STRUCT': list(range(0, max(1, q1))),
            'MESO':   list(range(max(1, q1), max(2, q2))),
            'MICRO':  list(range(max(2, q2), max(3, q3))),
            'DETAIL': list(range(max(3, q3), M)),
        }
    
    def inject_impulse(self, impulses: np.ndarray):
        """
        Inject complex impulses into nodes.
        
        impulses: (N,) complex array — flow_magnitude * exp(i * flow_angle)
                  at each crystal node position.
        """
        self.z += impulses
    
    def step(self, n_steps: int = 3):
        """
        Evolve the reservoir for n_steps, then project onto eigenmodes.
        
        Returns:
            band_gains: dict {STRUCT, MESO, MICRO, DETAIL} → float in [0, 1]
            mode_amps:  (M,) array of eigenmode amplitudes
        """
        for _ in range(n_steps):
            new_z = np.copy(self.z)
            
            for k in range(self.N):
                # Diffusion from neighbors
                diffusion = 0.0 + 0j
                for j, w in self.neighbors[k]:
                    diffusion += w * (self.z[j] - self.z[k])
                
                # Self-trapping
                amp_sq = abs(self.z[k]) ** 2
                trap = -self.beta * amp_sq * self.z[k]
                
                # Clockfield metric
                gamma = 1.0 / (1.0 + self.tau * amp_sq) ** 2
                
                # Damping
                damp = -self.damping * self.z[k]
                
                # Update
                new_z[k] += self.dt * gamma * (
                    self.coupling * diffusion + trap + damp
                )
            
            self.z = new_z
        
        # Project onto eigenmodes: a_m = Σ_k z_k · v_m(k)
        raw_amps = np.zeros(self.M, dtype=np.float64)
        for m in range(self.M):
            projection = np.sum(self.z * self.topo.eigenvectors[:, m])
            raw_amps[m] = abs(projection) ** 2
        
        # Exponential moving average (persistence of vision)
        self.mode_amps = (1 - self.ema_alpha) * self.mode_amps + self.ema_alpha * raw_amps
        
        # Compute band gains
        band_gains = {}
        for band_name, mode_indices in self.band_ranges.items():
            if mode_indices:
                energy = sum(self.mode_amps[m] for m in mode_indices)
            else:
                energy = 0.0
            band_gains[band_name] = energy
        
        # Normalize gains to [0, 1] range
        max_gain = max(band_gains.values()) + 1e-10
        for band_name in band_gains:
            band_gains[band_name] = np.clip(band_gains[band_name] / max_gain, 0, 1)
        
        return band_gains, self.mode_amps.copy()
    
    def get_node_screen_positions(self, frame_width: int, frame_height: int) -> np.ndarray:
        """
        Map crystal node positions to screen coordinates.
        
        Returns (N, 2) array of (x, y) in pixel coordinates.
        """
        positions = np.zeros((self.N, 2))
        for k in range(self.N):
            # node_positions are (row, col) in [0, field_size)
            row, col = self.topo.node_positions[k]
            positions[k, 0] = col / self.topo.field_size * frame_width
            positions[k, 1] = row / self.topo.field_size * frame_height
        return positions
    
    def get_state_visualization(self) -> dict:
        """Return current state for visualization."""
        return {
            'node_magnitudes': np.abs(self.z),
            'node_phases': np.angle(self.z),
            'mode_amps': self.mode_amps.copy(),
            'eigenvalues': self.topo.eigenvalues.copy(),
        }


def grow_crystal(size: int = 48, beta: float = 0.8, steps: int = 300,
                 seed: int = 42, n_modes: int = 12) -> CrystalReservoir:
    """
    Convenience: grow a field, extract topology, create reservoir.
    
    Returns a ready-to-use CrystalReservoir.
    """
    grower = PhiFieldGrower(size=size, beta=beta, seed=seed)
    grower.evolve(steps=steps)
    topo = grower.extract_topology(n_modes=n_modes)
    reservoir = CrystalReservoir(topo)
    
    print(f"Crystal grown: {reservoir.N} nodes, {n_modes} eigenmodes")
    print(f"  Eigenvalues: {topo.eigenvalues[:6].round(4)}")
    print(f"  Band assignment: { {k: len(v) for k, v in reservoir.band_ranges.items()} }")
    
    return reservoir


if __name__ == "__main__":
    res = grow_crystal()
    
    # Test: inject random impulse
    impulse = 0.5 * np.exp(1j * np.random.uniform(0, 2*np.pi, res.N))
    res.inject_impulse(impulse)
    
    for t in range(20):
        gains, amps = res.step(n_steps=3)
        print(f"  t={t:3d}  STRUCT={gains['STRUCT']:.3f}  MESO={gains['MESO']:.3f}  "
              f"MICRO={gains['MICRO']:.3f}  DETAIL={gains['DETAIL']:.3f}")
