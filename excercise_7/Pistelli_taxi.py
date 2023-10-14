# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode
epsilon=0.1 # Exploration rate

# changed Q-table initialization because with negatuve values at the beginning it wasn't working
# Q tables for rewards
Q_reward = np.zeros((500,6)) # All same
#Q_reward = -1000*np.random.random((500, 6)) # Random values

# Define a function to choose an action based on the current state using an epsilon-greedy policy
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        action = env.action_space.sample()
    else:
        # Exploit: choose the best action from the Q-table
        action = np.argmax(Q_reward[state,:])
    return action

# Training w/ random sampling of actions
for i in range(num_of_episodes):
    state = env.reset()[0]
    done = False 

    for j in range(num_of_steps): #num of step to not make it run it forever
        action = choose_action(state)
        new_state, reward, done, truncated, info = env.step(action) 
        if done:
            Q_reward[state,action] = reward #update reward in the table
            break
        else:
            Q_reward[state, action] = Q_reward[state, action] + alpha * (reward + gamma * np.max(Q_reward[new_state, :]) - Q_reward[state, action]) #update reward in the table
            state = new_state

print("----------------------")
print("Training done!")
print("----------------------")

# Testing
total_rewards = 0
total_actions = 0
for i in range(10):
    state = env.reset()[0]
    done = False
    
    while not done:
        action = np.argmax(Q_reward[state,:])
        state, reward, done, truncated, info = env.step(action)
        
        total_rewards += reward
        total_actions += 1

# Compute average total reward and average number of actions
print("Average total reward: ", total_rewards / 10)
print("Average number of actions: ", total_actions / 10)

