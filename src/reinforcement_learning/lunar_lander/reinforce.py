import torch

def compute_returns(rewards, gamma=0.99):
    G, ret = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        ret.append(G)
    ret.reverse()
    ret = torch.tensor(ret, dtype=torch.float32)
    return (ret - ret.mean()) / (ret.std() + 1e-8)

def reinforce_update(optimizer, logprobs, returns):
    loss = -(torch.stack(logprobs) * returns).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
