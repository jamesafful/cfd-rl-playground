import numpy as np

class LinearPolicy:
    """
    A tiny "RL-like" policy: action = clip( w dot feat + b )
    Features: [1, temp_error, temp_error_prev, temp_error_integral]
    """
    def __init__(self, heater_max: float, w: np.ndarray, b: float, a_min=0.0, a_max=None):
        self.w = np.asarray(w, dtype=np.float32).reshape(-1)
        self.b = float(b)
        self.heater_max = float(heater_max)
        self.a_min = float(a_min)
        self.a_max = float(heater_max if a_max is None else a_max)
        self.integral = 0.0
        self.prev_err = 0.0
        self.initialized = False

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.initialized = False

    def __call__(self, temp_error: float) -> np.ndarray:
        if not self.initialized:
            self.prev_err = temp_error
            self.initialized = True
        self.integral += temp_error
        feat = np.array([1.0, temp_error, self.prev_err, self.integral], dtype=np.float32)
        a = float(np.dot(self.w, feat) + self.b)
        self.prev_err = temp_error
        a = np.clip(a, self.a_min, self.a_max)
        return np.array([a], dtype=np.float32)

def load_linear_policy(path: str, heater_max: float):
    data = np.load(path)
    w = data['w']
    b = float(data['b'])
    return LinearPolicy(heater_max=heater_max, w=w, b=b)
