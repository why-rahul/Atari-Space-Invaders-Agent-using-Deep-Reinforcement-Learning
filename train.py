import torch
import numpy as np

def train_dqn(policy_net, target_net, buffer, optimizer, batch_size, gamma, device):
    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    
    states = torch.from_numpy(np.array(states)).float().to(device)
    next_states = torch.from_numpy(np.array(next_states)).float().to(device)

    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions).squeeze()
    next_q = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q * (1 - dones)

    loss = torch.nn.functional.mse_loss(q_values, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
