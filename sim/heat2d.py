from dataclasses import dataclass
import numpy as np
from .numerics import laplacian_5pt, cfl_heat, clip_action

@dataclass
class HeatParams:
    alpha: float          # thermal diffusivity (m^2/s)
    T_ambient: float      # ambient boundary temperature (K)
    heater_max: float     # max heater power density (W/m^2)

class HeatSim:
    def __init__(self, Nx=64, Ny=64, Lx=1.0, Ly=1.0, dt=1e-3, 
                 params: HeatParams = None, heaters=None, bc_type: str='dirichlet', dtype=np.float32):
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.dx, self.dy = self.Lx/self.Nx, self.Ly/self.Ny
        self.dt = float(dt)
        self.dtype = dtype
        self.params = params or HeatParams(alpha=1e-3, T_ambient=300.0, heater_max=200.0)
        self.bc_type = bc_type
        # default heater: 10x10 in center
        if heaters is None:
            cx, cy = self.Nx//2, self.Ny//2
            h = 10
            heaters = [(slice(cy-h//2, cy+h//2), slice(cx-h//2, cx+h//2))]
        self.heaters = heaters
        self.T = np.full((self.Ny, self.Nx), self.params.T_ambient, dtype=self.dtype)
        self.rho_cp = 1.0  # simplified units (documented)
        # mask for heater regions for quick add
        self._heater_masks = []
        for ys, xs in self.heaters:
            m = np.zeros_like(self.T, dtype=self.dtype)
            m[ys, xs] = 1.0
            self._heater_masks.append(m)

    def reset(self, T0: np.ndarray | None=None):
        if T0 is None:
            self.T = np.full_like(self.T, self.params.T_ambient)
            self.T += (np.random.randn(*self.T.shape).astype(self.dtype)) * 0.1
        else:
            assert T0.shape == self.T.shape
            self.T[...] = T0.astype(self.dtype)
        return self.T

    def apply_bc(self, T: np.ndarray):
        if self.bc_type == 'dirichlet':
            T[0,:]  = self.params.T_ambient
            T[-1,:] = self.params.T_ambient
            T[:,0]  = self.params.T_ambient
            T[:,-1] = self.params.T_ambient
        elif self.bc_type == 'neumann':
            T[0,:]  = T[1,:]
            T[-1,:] = T[-2,:]
            T[:,0]  = T[:,1]
            T[:,-1] = T[:,-2]
        else:
            raise ValueError('Unknown bc_type: ' + str(self.bc_type))

    def inject_heat(self, T: np.ndarray, action: np.ndarray):
        # action is vector of length K; clamp to [0, heater_max]
        K = len(self._heater_masks)
        a = np.asarray(action).reshape(K).astype(self.dtype)
        a = clip_action(a, 0.0, self.params.heater_max)
        # add q/(rho*cp) scaled by dt to temperature change: dT = dt*(alpha*L + q/(rho*cp))
        # we implement q term directly in step using masks
        return a

    def step(self, action: np.ndarray):
        T = self.T
        alpha = self.params.alpha
        dx, dy = self.dx, self.dy
        dt = self.dt

        # Laplacian
        L = laplacian_5pt(T, dx, dy).astype(self.dtype)

        # heater source
        a = self.inject_heat(T, action)
        q_term = np.zeros_like(T, dtype=self.dtype)
        for k, m in enumerate(self._heater_masks):
            if a[k] != 0.0:
                q_term += m * (a[k] / self.rho_cp)

        # forward Euler
        T_new = T + dt * (alpha * L + q_term)

        # apply BC
        self.apply_bc(T_new)

        # CFL check & optional dt scaling (we don't change dt mid-step, just report/cap future loops in app)
        cfl = cfl_heat(alpha, dt, dx, dy)
        exploded = bool(np.any(~np.isfinite(T_new)))
        if not exploded:
            self.T[...] = T_new
        return {
            "T": self.T,
            "cfl": cfl,
            "dt": dt,
            "dt_scaled": False,
            "exploded": exploded
        }
