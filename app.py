import numpy as np
try:
    import gradio as gr
except Exception:
    gr = None

from sim.heat2d import HeatSim, HeatParams
from sim.viz import heat_frame_png, timeseries_plot
from baselines.pid_heat import HeatPIDController
from rl.rl_policy import load_linear_policy

def make_default_target_mask(Ny, Nx):
    mask = np.zeros((Ny, Nx), dtype=np.float32)
    ys = slice(Ny//2+4, Ny//2+14)
    xs = slice(Nx//2+8, Nx//2+18)
    mask[ys, xs] = 1.0
    return mask

def run_episode(grid_size=64, steps=300, seed=0, alpha=1e-3, dt=1e-3, T_ambient=300.0, target_T=310.0, 
                heater_max=200.0, bc_type='dirichlet', controller='PID', act_interval=4, policy_path='rl/policies/heat_linear.npz'):

    rng = np.random.default_rng(None if seed == -1 else int(seed))
    Ny = Nx = int(grid_size)
    params = HeatParams(alpha=float(alpha), T_ambient=float(T_ambient), heater_max=float(heater_max))
    sim = HeatSim(Nx=Nx, Ny=Ny, dt=float(dt), params=params, bc_type=bc_type)
    sim.reset()

    target_mask = make_default_target_mask(Ny, Nx).astype(np.float32)
    target_area = np.maximum(1.0, float(target_mask.sum()))

    # Controllers
    if controller == 'PID':
        ctrl = HeatPIDController(heater_max=heater_max, dt_ctrl=act_interval*dt, kp=500.0, ki=50.0, kd=0.0, n_heaters=len(sim._heater_masks), target_T=target_T)
    elif controller == 'RL-Linear':
        import os
        _pp = policy_path
        if not os.path.exists(_pp):
            _pp = os.path.join(os.path.dirname(__file__), _pp)
        ctrl = load_linear_policy(_pp, heater_max=heater_max)
    else:
        ctrl = None

    frames_png = []
    rewards = []
    actions = []
    temps = []
    times = []

    T_vmin = T_ambient - 5.0
    T_vmax = target_T + 10.0

    a = np.zeros((len(sim._heater_masks),), dtype=np.float32)
    for t in range(int(steps)):
        # control
        if (t % int(act_interval) == 0):
            region_mean = float((sim.T * target_mask).sum() / target_area)
            if controller == 'PID':
                a = ctrl(region_mean)
            elif controller == 'RL-Linear':
                temp_error = (target_T - region_mean)
                a = ctrl(temp_error)
            else:
                a = np.zeros((len(sim._heater_masks),), dtype=np.float32)
        # step sim
        info = sim.step(a)
        # reward (tracking - effort penalty)
        region_mean = float((sim.T * target_mask).sum() / target_area)
        err = region_mean - target_T
        r_track = - (err**2)
        r_effort = - 1e-4 * float((a*a).mean())
        r = r_track + r_effort
        rewards.append(r)
        actions.append(float(a.mean()))
        temps.append(region_mean)
        times.append(t)
        # frame
        if (t % 2) == 0 or t == steps-1:
            frames_png.append(heat_frame_png(sim.T, vmin=T_vmin, vmax=T_vmax, target_mask=target_mask))

        if info['exploded']:
            break

    reward_png = timeseries_plot(times, rewards, title='Reward', ylabel='r')
    action_png = timeseries_plot(times, actions, title='Action (mean power)', ylabel='W/m^2')
    temp_png = timeseries_plot(times, temps, title='Target Region Temperature', ylabel='K')

    summary = f"""
**Episode finished**  
Steps: {len(times)}  
Final region mean T: {temps[-1]:.2f} K (target {target_T} K)  
Mean action: {np.mean(actions):.2f} W/m²  
"""

    return frames_png[-1], reward_png, action_png, temp_png, summary

def _build_demo():
    if gr is None:
        raise RuntimeError('Gradio is not available. Install requirements and run: python app.py')
    with gr.Blocks() as demo:
        gr.Markdown("""# CFD Heat Control Playground
Control the 2D heat equation with a PID or a tiny RL-like linear policy. Adjust physics and run.
""")
        with gr.Row():
            with gr.Column(scale=1):
                grid_size = gr.Slider(48, 96, value=64, step=16, label='Grid size (Nx=Ny)')
                steps = gr.Slider(50, 800, value=300, step=10, label='Steps')
                seed = gr.Number(value=0, label='Seed (-1=random)')
                alpha = gr.Number(value=1e-3, label='alpha (m^2/s)')
                dt = gr.Number(value=1e-3, label='dt (s)')
                T_ambient = gr.Number(value=300.0, label='Ambient T (K)')
                target_T = gr.Number(value=310.0, label='Target T (K)')
                heater_max = gr.Number(value=200.0, label='Heater max (W/m^2)')
                bc_type = gr.Dropdown(['dirichlet', 'neumann'], value='dirichlet', label='Boundary condition')
                controller = gr.Dropdown(['PID', 'RL-Linear', 'None'], value='PID', label='Controller')
                act_interval = gr.Slider(1, 10, value=4, step=1, label='Control interval (steps)')
                run_btn = gr.Button('Run')
            with gr.Column(scale=2):
                frame = gr.Image(label='Field (T)')
                reward = gr.Image(label='Reward')
                action = gr.Image(label='Action')
                temp = gr.Image(label='Target Region T')
                summary = gr.Markdown()
        run_btn.click(run_episode, 
                      inputs=[grid_size, steps, seed, alpha, dt, T_ambient, target_T, heater_max, bc_type, controller, act_interval],
                      outputs=[frame, reward, action, temp, summary])
    return demo

if __name__ == '__main__':
    demo = _build_demo()
    demo.launch()
