import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent import ACAgent
from brimarl_masked.algorithms.algorithm import Algorithm


class A2CAlgorithm(Algorithm):
    def __init__(self, num_players, discount, num_learning_per_epoch, min_samples=128, entropy_beta=0.0, epsilon=1e-8):
        self.num_players = num_players
        self.discount = discount
        self.optimizer_actor = tf.optimizers.legacy.Adam(1e-4)
        self.optimizer_critic = tf.optimizers.legacy.Adam(3e-4)
        self.num_learning_per_epoch = num_learning_per_epoch
        self.entropy_beta = entropy_beta
        self.epsilon = epsilon
        self.min_samples = min_samples

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []

    """
    states: List of shape (N, S) where N is the number of transitions and S the state space
    actions: List of shape (N, 1)
    rewards: List of shape (N, 1)
    dones: List of shape (N, 1)
    """

    def store_game(self, states, actions, masks, rewards, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        masks = np.array(masks)

        next_states = np.copy(states)[self.num_players:]
        dones = dones[self.num_players:]

        states = states[:-self.num_players]
        actions = actions[:-self.num_players]
        rewards = rewards[:-self.num_players]
        masks = masks[:-self.num_players]

        for i in range(0, len(states), self.num_players):
            self.s.append(states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]])
            self.sn.append(next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]])
            self.a.append(actions[i])
            self.r.append(rewards[i])
            self.d.append(dones[i])
            self.m.append(masks[i])

    def learn(self, agent: ACAgent):
        if len(self.s) < self.min_samples:
            return None

        self.s = np.array(self.s).squeeze()
        self.a = np.array(self.a)
        self.r = np.array(self.r)[:, None]
        self.sn = np.array(self.sn).squeeze()
        self.d = np.array(self.d)[:, None]
        self.m = np.array(self.m)
        BS = self.s.shape[0]

        self.assert_same_shape(self.s, (BS, self.s.shape[1]))
        self.assert_same_shape(self.a, (BS, self.a.shape[1]))
        self.assert_same_shape(self.r, (BS, 1))
        self.assert_same_shape(self.sn, self.s)
        self.assert_same_shape(self.d, (BS, 1))
        self.assert_same_shape(self.m, self.a)

        loss = 0.
        iterations = 0.
        for _ in range(self.num_learning_per_epoch):

            loss += loss_critic
            iterations += 1

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []
        return loss / (iterations + 1e-10) # in case something goes wrong and we divide by 0
