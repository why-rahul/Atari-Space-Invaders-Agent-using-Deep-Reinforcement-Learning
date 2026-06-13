import torch
import random
import numpy as np
from env_wrapper import AtariWrapper
from dqn_model import DQN
from replay_buffer import ReplayBuffer
from train import train_dqn


ENV_NAME = "ALE/SpaceInvaders-v5"
EPISODES = 130
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPSILON = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = AtariWrapper(ENV_NAME)
action_size = env.env.action_space.n

policy_net = DQN(action_size).to(device)
target_net = DQN(action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer()
episode_rewards = []
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if random.random() < EPSILON:
            action = random.randrange(action_size)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(state_t).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train_dqn(policy_net, target_net, buffer, optimizer, BATCH_SIZE, GAMMA, device)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    print(f"Episode {episode}, Reward: {total_reward}")
    episode_rewards.append(total_reward)
torch.save(policy_net.state_dict(), "space_invaders_dqn.pth")
env.close()

import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training Reward Curve - Space Invaders (DQN)")
plt.show()

