import itertools
import random
import sys

import gym
import matplotlib
import numpy as np
from lib import plotting

env = gym.envs.make('FrozenLake-v0')

Q = np.zeros([16, 4])


def setup_agent(state_size, action_size):
    gamma = 0.95  # Discounting rate
    learning_rate = 0.8

    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = (epsilon_start - epsilon_min) / 50000
    curr_epsilon = epsilon_start

    def update_epsilon():
        nonlocal curr_epsilon
        if curr_epsilon > epsilon_min:
            curr_epsilon -= epsilon_decay

    def should_explore_fn():
        return np.random.rand() <= curr_epsilon

    def policy(state, explore_fn=should_explore_fn):
        if explore_fn():
            return random.randrange(action_size)
        return np.argmax(Q[state, :])

    def update_agent(state, action, reward, next_state, done):
        Q[state, action] = Q[state, action] + learning_rate * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        update_epsilon()
        return curr_epsilon

    return policy, update_agent


policy_fn, update_agent_fn = setup_agent(16, 4)


def learning(env, policy_fn, update_agent_fn, num_episodes, render=False):
    rewards = []
    for i_episode in range(num_episodes):

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


r = learning(env, policy_fn, update_agent_fn, 5000)

env.reset()
learning(env, policy_fn, update_agent_fn, 5, True)
env.close()

import matplotlib.pyplot as plt

plt.plot(np.array(r))
plt.ylabel('reward over time')
plt.show()
