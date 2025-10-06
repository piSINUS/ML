import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

alpha = 0.1        
gamma = 0.95       
epsilon = 0.1 
episodes = 10000   
max_steps = 200

state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))

for ep in range(episodes):
    state, _ = env.reset()
    for _ in range(max_steps):
        
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        if done:
            break

policy = np.argmax(Q, axis=1)

wins = 0
rewards = []

for ep in range(10000):
    state, _ = env.reset()
    total_reward = 0
    for step in range(200):
        action = policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            if reward == 20:
                wins += 1
            break
    rewards.append(total_reward)

print(f"Средняя награда: {np.mean(rewards):.2f}")
print(f"Побед (пассажир доставлен): {wins}")

env = gym.make("Taxi-v3", render_mode="ansi")
state, _ = env.reset()
print(env.render())

for _ in range(50):
    action = policy[state]
    state, reward, terminated, truncated, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    print(env.render())
    if terminated or truncated:
        break

