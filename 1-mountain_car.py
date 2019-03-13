import itertools
import sys

import gym
import matplotlib
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing

if "./lib" not in sys.path:
    sys.path.append("./lib")

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')
env = gym.envs.make("MountainCar-v0")


class SGDAgent:
    """
    Value Function approximator.
    """

    def __init__(self, initial_state, action_space, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
        self.models = []
        self.feature_state_fn = setup_scaler()

        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space

        self.initialize_model(initial_state)

    def get_action(self, state, i_episode):

        def policy_fn(q_values, i_episode):
            curr_epsilon = epsilon * epsilon_decay ** i_episode

            best_action = np.argmax(q_values)
            selected_action = best_action

            if curr_epsilon > 0:
                action_probs = np.ones(action_space, dtype=float) * curr_epsilon / action_space
                action_probs[selected_action] += (1.0 - curr_epsilon)
                selected_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            return selected_action

        state = self.feature_state_fn(state)
        q_values = self.predict(state)
        return self.policy_fn(q_values, i_episode)

    def predict(self, state, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        if not a:
            return np.array([m.predict(state)[0] for m in self.models])
        else:
            return self.models[a].predict(state)[0]

    def update(self, state, next_state, action, reward, done):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        # TD Update
        q_values_next = self.predict(next_state)
        td_target = reward + self.discount_factor * np.max(q_values_next)

        state = self.feature_state_fn(state)
        self.models[action].partial_fit(state, [reward])

    def initialize_model(self, initial_state):
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit(self.feature_state_fn(initial_state), [0])
            self.models.append(model)


def setup_scaler():
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurized representation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    ])
    featurizer.fit(scaler.transform(observation_examples))

    def featurize_state(state):
        """
        Returns the featurized representation for a state.
        """
        scaled = scaler.transform([state])
        featurized = featurizer.transform(scaled)
        return [featurized[0]]

    return featurize_state


def learning(env, agent: SGDAgent, num_episodes):
    """
    Q-Learning algorithm for fff-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        state = env.reset()

        for t in itertools.count():

            action = agent.get_action(state,  i_episode)

            next_state, reward, done, _ = env.step(action)

            agent.update(state, next_state, action, reward, done)

            state = next_state

            print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, last_reward), end="")

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            if done:
                break

    return stats


state = env.reset()
agent = SGDAgent(state, env.action_space.n)
stats = learning(env, agent, 180)

state = env.reset()
for t in range(100):
    env.render()

    action = agent.get_action(state)
    # action = env.action_space.sample()

    state, reward, done, info = env.step(action)

    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
