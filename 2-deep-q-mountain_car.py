import gym
import itertools
import random
import numpy as np
import sys

import matplotlib

# matplotlib.use("TkAgg")

matplotlib.style.use('ggplot')
env = gym.envs.make("MountainCar-v0")

from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.0

        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=10000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        state = self.reformat_state(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

        return action

    def append_to_replay_memory(self, state, action, reward, next_state, done):
        state = self.reformat_state(state)
        next_state = self.reformat_state(next_state)
        self.memory.append((state, action, reward, next_state, done))
        # print(len(self.memory))

    def update_epsilon(self):
        epsilon_start = 1.0
        epsilon_min = 0.005
        epsilon_decay = (epsilon_start - epsilon_min) / 50000

        if self.epsilon > epsilon_min:
            self.epsilon -= epsilon_decay

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * np.amax(self.target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

    def reformat_state(self, state):
        return np.reshape(state, [1, self.state_size])


def setup_take_action(agent: DQNAgent, frequency: int):
    """
        takes action only once for the given number of steps
    :param frequency:
    :type agent: DQNAgent
    """
    counter = 0

    def take_action_fn(state, step_counter):
        nonlocal counter
        counter = counter + 1
        action = 1  # no push action
        if counter % frequency:
            action = agent.get_action(state, step_counter)
        return action

    return take_action_fn


def learning(env, agent: DQNAgent, num_episodes, action_frequency):
    take_action_fn = setup_take_action(agent, action_frequency)

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
    state_size = env.observation_space.shape[0]

    for i_episode in range(num_episodes):

        sys.stdout.flush()
        state = env.reset()

        fake_action = 0
        action_count = 0

        sum = 0

        for t in itertools.count():

            action_count = action_count + 1

            if action_count == 4:
                fake_action = agent.get_action(state)
                action_count = 0

            next_state, reward, done, _ = env.step(fake_action)

            agent.append_to_replay_memory(state, fake_action, reward, next_state, done)
            agent.update_epsilon()

            sum += reward
            agent.train_replay()

            state = next_state

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if i_episode % 50 == 0:
                # agent.save_model("./save_model/mountain_car_dqn_my.h5")
                pass

            if done:
                reward_sum = stats.episode_rewards[i_episode]
                print("\nStep {} @ Episode {}/{} ({}) eps. {}".format(t, i_episode + 1, num_episodes, sum, agent.epsilon), end="")
                agent.update_target_model()
                break
    return stats


agent = DQNAgent(env.observation_space.shape[0], 3)
# agent.load_model("./save_model/mountain_car_dqn.h5")
learning(env, agent, 200, 4)

''' 
observation = env.reset()
for t in range(100):
    env.render()
    action = policy_fn(observation)
    #action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

'''
