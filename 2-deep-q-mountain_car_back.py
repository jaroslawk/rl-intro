import itertools
import random
import sys

import gym
import matplotlib
import numpy as np

from lib import plotting

# matplotlib.use("TkAgg")

if "./lib" not in sys.path:
    sys.path.append("./lib")

matplotlib.style.use('ggplot')
env = gym.envs.make("MountainCar-v0")

from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


def reformat_state(state, state_size):
    return np.reshape(state, [1, state_size])


def build_model(learning_rate, state_size, action_size):
    model = Sequential()
    model.add(Dense(32, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def setup_agent(state_size, action_size):
    discount_factor = 0.99
    learning_rate = 0.001

    batch_size = 64
    train_start = 1000
    memory = deque(maxlen=10000)

    epsilon_start = 1.0
    epsilon_min = 0.005
    epsilon_decay = (epsilon_start - epsilon_min) / 50000
    curr_epsilon = epsilon_start

    model = build_model(learning_rate, state_size, action_size)
    target_model = build_model(learning_rate, state_size, action_size)

    def policy(state):

        if should_explore_fn():
            return random.randrange(action_size)

        state = reformat_state(state, state_size)
        q_value = model.predict(state)
        return np.argmax(q_value[0])

    def update_target_model():
        target_model.set_weights(model.get_weights())

    def update_agent(state, action, reward, next_state, done):

        state = reformat_state(state, state_size)
        next_state = reformat_state(next_state, state_size)

        memory.append((state, action, reward, next_state, done))

        update_epsilon()

        if len(memory) > train_start:
            train_replay()

        if done:
            update_target_model()

        return curr_epsilon

    def should_explore_fn():
        return np.random.rand() <= curr_epsilon

    def update_epsilon():
        nonlocal curr_epsilon
        if curr_epsilon > epsilon_min:
            curr_epsilon -= epsilon_decay

    def train_replay():

        curr_batch_size = min(batch_size, len(memory))
        mini_batch = random.sample(memory, batch_size)

        update_input = np.zeros((curr_batch_size, state_size))
        update_target = np.zeros((curr_batch_size, action_size))

        for i in range(curr_batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + discount_factor * np.amax(target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    update_target_model()

    return policy, update_agent


def learning(env, num_episodes):

    policy_fn, update_agent_fn = setup_agent(env.observation_space.shape[0], 3)

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        sys.stdout.flush()
        state = env.reset()

        s = 0
        for t in itertools.count():

            action = 1  # no action

            if t % 4 == 0:  # take action once 1/4 times
                action = policy_fn(state)

            next_state, reward, done, _ = env.step(action)

            curr_epsilon = update_agent_fn(state, action, reward, next_state, done)

            s += reward

            state = next_state

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if i_episode % 50 == 0:
                # agent.save_model("./save_model/mountain_car_dqn_my.h5")
                pass

            if done:
                reward_sum = stats.episode_rewards[i_episode]
                print("\nStep {} @ Episode {}/{} ({}) eps. {}".format(t, i_episode + 1, num_episodes, s, curr_epsilon), end="")
                break

    return stats


# agent.load_model("./save_model/mountain_car_dqn.h5")
learning(env, 300)

''' 


    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

'''
