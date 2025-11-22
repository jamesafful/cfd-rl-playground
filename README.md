# CFD Heat Control Playground (Hugging Face Space-ready)

An interactive **2D heat equation** simulator with a **PID baseline** and a tiny **RL-like linear policy**.
Runs on CPU and is designed to work out-of-the-box in Codespaces or Hugging Face Spaces.

## Quickstart

```bash
pip install -r requirements.txt
python app.py
```

Open the Gradio link and press **Run**.

## What it does

We solve the heat equation on a uniform grid with explicit Euler:

\[ \frac{\partial T}{\partial t} = \alpha \nabla^2 T + q(x,y,t) \]

- **BCs:** Dirichlet (fixed to ambient) or Neumann (insulated).
- **Control:** one or more heater patches add a source term `q` (W/m²). Here we ship a single centered heater.
- **Reward (for plots):** tracks target temperature in a small region (upper-right box) and penalizes power.

### Controllers

- **PID:** tuned conservatively; tries to keep the target region at `target_T`.
- **RL-Linear:** a tiny linear policy on error features (ships with weights in `rl/policies/heat_linear.npz`).

> Note: You can replace the linear policy with your own or extend to Stable-Baselines3.
> This repo intentionally keeps dependencies light for Spaces.

## Files

- `app.py`: Gradio UI.
- `sim/heat2d.py`: heat simulator.
- `sim/numerics.py`: Laplacian stencil, CFL, action clipping.
- `sim/viz.py`: PNG plots for fields and time series.
- `baselines/pid_heat.py`: PID controller.
- `rl/rl_policy.py`: simple linear-policy loader; weights in `rl/policies/heat_linear.npz`.

## Limits

- Simplified units (`rho*cp=1`) for clarity.
- Coarse grids & explicit stepping; choose small `dt` for stability (`cfl_heat <= ~0.25` is safe).
- Only heat equation (no Navier–Stokes) in this minimal working version.

## License

MIT
