import gymnasium as gym, torch, torch.nn.functional as F
from pathlib import Path
from .models import PolicyNet
from .utils import save_gif

def play(episodes=5, env_id="LunarLander-v3", render_mode="human"):
    env = gym.make(env_id, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNet(obs_dim, act_dim)
    ckpt = Path(__file__).resolve().parents[3] / "docs" / "lunarlander_policy.pt"
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = logits.argmax(dim=-1).item()
            obs, _, done, trunc, _ = env.step(action)
    env.close()
    
def rollout_frames(episodes=1, env_id="LunarLander-v3", policy_path=None, fps=30):
    env = gym.make(env_id, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]; act_dim = env.action_space.n
    policy = PolicyNet(obs_dim, act_dim)
    if policy_path:
        policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy.eval()

    all_frames = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = trunc = False
        while not (done or trunc):
            logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = logits.argmax(dim=-1).item()
            obs, _, done, trunc, _ = env.step(action)
            frame = env.render()
            all_frames.append(frame)
    env.close()
    root = Path(__file__).resolve().parents[3]
    gif_path = root / "docs" / "gifs" / "lunarlander_demo.gif"
    save_gif(all_frames, gif_path, fps=fps)
    return gif_path

if __name__ == "__main__":
    play()
