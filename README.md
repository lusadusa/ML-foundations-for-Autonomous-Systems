# ML-foundations-for-Autonomous-Systems
This repository collects my work while transitioning from aerospace engineering (GNC/AOCS) into AI, Machine Learning, and Autonomous Systems.   It includes implementations of ML algorithms, neural networks, and reinforcement learning applied to dynamic systems.

## Structure
src/{regression,classification,clustering,neural_networks,reinforcement_learning}
docs/ (plots,gifs) · notebooks/ (demos) · tests/ · requirements.txt

## Projects
### RL – LunarLander (REINFORCE)
Goal: learn to land in OpenAI Gymnasium.
Approach: policy gradient (REINFORCE)
Result: avg reward ≈ 334 (target > 200) after 600 episodes.
How to run:
```bash
pip install -r requirements.txt
python -m src.reinforcement_learning.lunar_lander.train
