import os, random, numpy as np, torch, matplotlib.pyplot as plt

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
