import io
import numpy as np
import matplotlib.pyplot as plt

def heat_frame_png(T: np.ndarray, vmin=None, vmax=None, target_mask: np.ndarray|None=None) -> bytes:
    fig, ax = plt.subplots(figsize=(4,4), dpi=100)
    im = ax.imshow(T, origin='lower', vmin=vmin, vmax=vmax)
    if target_mask is not None:
        # overlay contour of target region
        ax.contour(target_mask, levels=[0.5], linewidths=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def timeseries_plot(xs, ys, title: str, ylabel: str) -> bytes:
    fig, ax = plt.subplots(figsize=(4,2), dpi=120)
    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel('t (steps)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
