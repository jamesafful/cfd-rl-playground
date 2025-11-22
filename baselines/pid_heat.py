import numpy as np

class PID:
    def __init__(self, kp: float, ki: float, kd: float, umin: float, umax: float, dt_ctrl: float):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.umin, self.umax = umin, umax
        self.dt = dt_ctrl
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
        self.initialized = False

    def act(self, error: float) -> float:
        if not self.initialized:
            self.prev_err = error
            self.initialized = True
        # PID
        self.integral += error * self.dt
        deriv = (error - self.prev_err) / self.dt if self.dt > 0 else 0.0
        u = self.kp*error + self.ki*self.integral + self.kd*deriv
        # clamp & anti-windup
        if u > self.umax:
            u = self.umax
            self.integral -= error * self.dt * 0.5
        elif u < self.umin:
            u = self.umin
            self.integral -= error * self.dt * 0.5
        self.prev_err = error
        return u

class HeatPIDController:
    def __init__(self, heater_max: float, dt_ctrl: float, kp=500.0, ki=50.0, kd=0.0, n_heaters=1, target_T=310.0):
        self.pid = PID(kp, ki, kd, 0.0, heater_max, dt_ctrl)
        self.n_heaters = n_heaters
        self.target_T = target_T
        self.heater_max = heater_max

    def reset(self):
        self.pid.reset()

    def __call__(self, current_region_T_mean: float) -> np.ndarray:
        err = self.target_T - float(current_region_T_mean)
        u = self.pid.act(err)
        return np.full((self.n_heaters,), u, dtype=np.float32)
