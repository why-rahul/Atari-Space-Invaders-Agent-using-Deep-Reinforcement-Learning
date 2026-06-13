# Deep Reinforcement Learning for Atari Space Invaders using DQN

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to play the Atari Space Invaders game through reinforcement learning. The agent interacts with the environment, receives rewards, and improves its policy over time using deep neural networks.

The project is built using Gymnasium and the Arcade Learning Environment (ALE).

---

## Features

* Deep Reinforcement Learning using DQN
* Atari Space Invaders environment
* Frame preprocessing and frame stacking
* Experience Replay Buffer
* Target Network for stable training
* CNN-based Q-value approximation
* Autonomous gameplay after training

---

## Environment

* Environment: ALE/SpaceInvaders-v5
* Framework: Gymnasium
* Emulator: Arcade Learning Environment (ALE)

---

## State Representation

* RGB frames converted to grayscale
* Frames resized to 84 × 84
* Four consecutive frames stacked
* Final state dimension: 84 × 84 × 4

---

## Action Space

The agent can perform the following actions:

* Move Left
* Move Right
* Fire

---

## Deep Q-Network Architecture

The agent uses a Convolutional Neural Network (CNN) to approximate Q-values directly from visual game frames.

The Bellman target is used to train the network:

y = r + γ max Q(s', a')

The network minimizes the difference between predicted and target Q-values using gradient descent.

---

## Training

* Algorithm: Deep Q-Network (DQN)
* Training Episodes: 500
* Exploration Strategy: Epsilon-Greedy
* Optimizer: Adam

---

## Results

The agent learned basic movement and shooting behavior through interaction with the environment.


---

## Project Structure

main.py - Training loop

env_wrapper.py - Environment preprocessing and frame stacking

dqn_model.py - Deep Q-Network architecture

replay_buffer.py - Experience replay memory

train.py - DQN training procedure

test_agent.py - Evaluation of trained agent

space_invaders_dqn.pth - Trained model weights

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python main.py
```

---

## Test Trained Agent

```bash
python test_agent.py
```

---

## Author

Rahul Jana

M.Sc. Computer Science
