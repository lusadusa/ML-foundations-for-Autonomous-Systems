import os, random, numpy as np, torch, matplotlib.pyplot as plt
import imageio.v2 as imageio

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def plot_rewards(rewards, outpath):
    import numpy as np
    x = np.arange(len(rewards))
    y = np.array(rewards, dtype=float)
    fig = plt.figure()
    plt.plot(x, y, linewidth=1.0)
    plt.xlabel("Episode"); plt.ylabel("Return")
    plt.title("LunarLander – REINFORCE")
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def save_gif(frames, outpath, fps=30):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(outpath, frames, fps=fps)