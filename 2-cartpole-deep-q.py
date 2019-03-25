import itertools
import random
from collections import deque

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

matplotlib.style.use('ggplot')


def build_model(state_size, action_size, learning_rate=0.007):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model


def setup_agent(action_size, epsilon_decay_rate=0.998, epsilon_start=1.0, epsilon_min=0.01):
    curr_epsilon = epsilon_start

    memory = deque(maxlen=10000)

    def update_epsilon():
        nonlocal curr_epsilon
        curr_epsilon *= epsilon_decay_rate
        curr_epsilon = max(epsilon_min, curr_epsilon)

    def train_replay(batch_size=65, discount_factor=0.95):

        if len(memory) < batch_size:
            return

        batch = random.sample(memory, batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + discount_factor * np.amax(model.predict(state_next)[0]))
            q_values = model.predict(state)
            q_values[0][action] = q_update
            model.fit(state, q_values, verbose=0)

    def policy_fn(state):
        if np.random.rand() <= curr_epsilon:
            return random.randrange(action_size)
        q = model.predict(state)
        return np.argmax(q[0])

    def update_fn(state, action, reward, next_state, done):
        memory.append((state, action, reward, next_state, done))

        train_replay()
        update_epsilon()

        return curr_epsilon

    return policy_fn, update_fn


def learning(env, policy_fn, update_agent_fn, num_episodes, render=False):

    def reformat_state(state):
        return np.reshape(state, [1, state_size])

    rewards = []
    for ep in range(num_episodes):

        state = reformat_state(env.reset())
        reward_sum = 0

        for st in itertools.count():

            if render:
                env.render()

            action = policy_fn(state)

            next_state, reward, done, _ = env.step(action)
            next_state = reformat_state(next_state)

            reward = reward if not done else -reward

            curr_epsilon = update_agent_fn(state, action, reward, next_state, done)

            state = next_state
            reward_sum += reward

            if done:
                print(' Episode:', ep, 'Epsilon:', curr_epsilon, ' Reward sum: ', reward_sum, ' score:', st)
                rewards.append(reward_sum)
                break

        if ep % 50 == 0:
            model.save_weights('carpool_0.pkl')

    return rewards


environment = gym.envs.make('CartPole-v1')
state_size = environment.observation_space.shape[0]

model = build_model(state_size, environment.action_space.n, state_size)
# model.load_weights('carpool2.pkl')
policy, update_agent = setup_agent(environment.action_space.n)
rewards = learning(environment, policy, update_agent, 120, False)

# lets play using learned model
# environment.reset()
# learning(environment, policy, update_agent, 5, True)
# environment.close()

plt.plot(np.array(rewards))
plt.ylabel('reward')
plt.xlabel('episode played')
plt.show()
1==1