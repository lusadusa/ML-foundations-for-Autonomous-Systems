import gymnasium as gym, torch, torch.nn.functional as F
from pathlib import Path
from .models import PolicyNet

def play(episodes=5):
    env = gym.make("LunarLander-v2", render_mode="human")
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

if __name__ == "__main__":
    play()
