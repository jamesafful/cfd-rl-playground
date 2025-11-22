# Model Card: Heat Control Linear Policy

- **Type**: Linear policy over temperature error features
- **Intended Use**: Educational control demo for 2D heat equation
- **Inputs**: Temperature error (current, previous, integral)
- **Outputs**: Heater command (W/m²), clipped to [0, heater_max]

Limitations: Not a general RL solution. For research-quality RL, replace with a PPO policy.
