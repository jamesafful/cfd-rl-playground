import numpy as np

def laplacian_5pt(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    # assume field shape (H, W) with y,x indexing
    H, W = field.shape
    out = np.zeros_like(field, dtype=field.dtype)
    inv_dx2 = 1.0 / (dx*dx)
    inv_dy2 = 1.0 / (dy*dy)
    # interior
    out[1:-1,1:-1] = (
        (field[1:-1,2:] - 2*field[1:-1,1:-1] + field[1:-1,0:-2]) * inv_dx2 +
        (field[2:,1:-1] - 2*field[1:-1,1:-1] + field[0:-2,1:-1]) * inv_dy2
    )
    # boundaries left/right: one-sided second derivative (Neumann approx); caller may override BC later
    out[:,0] = out[:,1]
    out[:,-1] = out[:,-2]
    out[0,:] = out[1,:]
    out[-1,:] = out[-2,:]
    return out

def cfl_heat(alpha: float, dt: float, dx: float, dy: float) -> float:
    return dt * 2*alpha * (1.0/(dx*dx) + 1.0/(dy*dy))

def clip_action(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(a, low), high)
