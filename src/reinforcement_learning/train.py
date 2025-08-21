import gymnasium as gym, torch, torch.nn.functional as F
from pathlib import Path
from .models import PolicyNet
from .reinforce import compute_returns, reinforce_update
from .utils import seed_all, plot_rewards

def train(episodes=1000, gamma=0.99, lr=3e-4, render_every=None, seed=42):
    seed_all(seed)
    env = gym.make("LunarLander-v2")  # requires gymnasium[box2d]
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    returns_hist = []
    for ep in range(episodes):
        logprobs, rewards = [], []
        obs, _ = env.reset(seed=seed + ep)
        done, trunc = False, False
        while not (done or trunc):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_t)
            probs  = F.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            logprobs.append(dist.log_prob(action))
            obs, reward, done, trunc, _ = env.step(action.item())
            rewards.append(reward)

        ret = compute_returns(rewards, gamma)
        loss = reinforce_update(optimizer, logprobs, ret)
        returns_hist.append(sum(rewards))

        if (ep+1) % 10 == 0:
            print(f"[{ep+1}/{episodes}] return={returns_hist[-1]:.1f} loss={loss:.3f}")

    env.close()
    root = Path(__file__).resolve().parents[3]
    out = root / "docs" / "images" / "lunarlander_rewards.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_rewards(returns_hist, out)
    torch.save(policy.state_dict(), root / "docs" / "lunarlander_policy.pt")
    return returns_hist

if __name__ == "__main__":
    train(episodes=500)
