import torch
from env_wrapper import AtariWrapper
from dqn_model import DQN

env = AtariWrapper("ALE/SpaceInvaders-v5", render=True)

action_size = env.env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(action_size).to(device)
model.load_state_dict(torch.load("space_invaders_dqn.pth"))
model.eval()

state, _ = env.reset()

total_score = 0
done = False
while not done:
    with torch.no_grad():
       state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
       action = model(state_t).argmax().item()

    state, reward, terminated, truncated, _ = env.step(action)
    total_score += reward
    done = terminated or truncated

print(f"Total Score: {total_score}")
env.close()
