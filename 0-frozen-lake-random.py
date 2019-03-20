import itertools
import random

import gym
import numpy as np

import matplotlib.pyplot as plt

state_size = 16
action_size = 4


def setup_agent():
    def policy_fn(state):
        return random.randrange(action_size)

    def update_fn(state, action, reward, next_state, done):
        # learning goes here
        pass

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
            update_agent_fn(state, action, reward, next_state, done)

            state = next_state
            reward_sum += reward
            if done:
                if reward_sum != 0:
                    print('Reward sum: ', reward_sum)
                rewards.append(reward_sum)
                break

    return rewards


environment = gym.envs.make('FrozenLake-v0')

policy, update_agent = setup_agent()
rewards = learning(environment, policy, update_agent, 5000)

plt.plot(np.array(rewards))
plt.ylabel('reward')
plt.xlabel('episode played')
plt.show()
