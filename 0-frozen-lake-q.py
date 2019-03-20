import itertools
import random

import gym
import numpy as np

import matplotlib.pyplot as plt

state_size = 16
action_size = 4

gamma = 0.95  # Discounting rate
learning_rate = 0.8

epsilon_start = 1.0
epsilon_min = 0.01

Q = np.zeros([state_size, action_size])


def setup_agent():
    epsilon_decay = (epsilon_start - epsilon_min) / 50000
    curr_epsilon = epsilon_start

    def update_epsilon():
        nonlocal curr_epsilon
        if curr_epsilon > epsilon_min:
            curr_epsilon -= epsilon_decay

    def should_explore():
        return np.random.rand() <= curr_epsilon

    def policy_fn(state):
        if should_explore():
            return random.randrange(action_size)
        return np.argmax(Q[state, :])

    def update_fn(state, action, reward, next_state, done):
        Q[state, action] = Q[state, action] + learning_rate * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        update_epsilon()
        return curr_epsilon

    return policy_fn, update_fn


def learning(env, policy_fn, update_agent_fn, num_episodes, render=False):
    rewards = []
    for _ in range(num_episodes):

        state = env.reset()
        reward_sum = 0
        for _ in itertools.count():

            if render:
                env.render()

            action = policy_fn(state)

            next_state, reward, done, _ = env.step(action)
            curr_epsilon = update_agent_fn(state, action, reward, next_state, done)

            state = next_state
            reward_sum += reward
            if done:
                if reward_sum != 0:
                    print('Epsilon:', curr_epsilon, ' Reward sum: ', reward_sum)
                rewards.append(reward_sum)
                break

    return rewards


environment = gym.envs.make('FrozenLake-v0')

policy, update_agent = setup_agent()
rewards = learning(environment, policy, update_agent, 5000)

# lets play using Q values
environment.reset()
learning(environment, policy, update_agent, 5, True)
environment.close()

plt.plot(np.array(rewards))
plt.ylabel('reward')
plt.xlabel('episode played')
plt.show()
