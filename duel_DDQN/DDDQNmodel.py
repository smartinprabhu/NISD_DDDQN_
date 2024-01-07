import gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dueling_ddqn_torch import DuelingDeepQNetwork, ReplayBuffer

# Define Constants
env = 'output.mp4'
observation_space = 'output.mp4'
# Replace 'UAVDT-v0' with the name of your custom UAVDT environment
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.001
TARGET_UPDATE = 1000
MEMORY_SIZE = 1000000
LR = 5e-4
NUM_EPISODES = 500


if __name__ == '__main__':
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DuelingDeepQNetwork(lr=LR, n_actions=n_actions, input_dims=[input_dims],
                                name='uavdt_dueling_ddqn', chkpt_dir='models/')
    target_network = DuelingDeepQNetwork(lr=LR, n_actions=n_actions, input_dims=[input_dims],
                                         name='uavdt_dueling_ddqn_target', chkpt_dir='models/')

    memory = ReplayBuffer(MEMORY_SIZE, input_dims=[input_dims])

    # Initialize epsilon for epsilon-greedy exploration
    epsilon = EPSILON_START

    scores = []
    eps_history = []

    for episode in range(NUM_EPISODES):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            if np.random.random() > epsilon:
                state = T.tensor([observation], dtype=T.float).to(agent.device)
                _, advantage = agent.forward(state)
                action = T.argmax(advantage).item()
            else:
                action = np.random.choice(n_actions)

            next_observation, reward, done, _ = env.step(action)
            score += reward

            memory.store_transition(observation, action, reward, next_observation, done)
            observation = next_observation

            agent.learn(memory, BATCH_SIZE, GAMMA, target_network)

        scores.append(score)
        eps_history.append(epsilon)
        avg_score = np.mean(scores[-100:])
        print(f"Episode: {episode}, Score: {score}, Average Score (last 100 episodes): {avg_score}")

        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(agent.state_dict())

        epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

    # Save the trained model
    agent.save_checkpoint()
    target_network.save_checkpoint()
