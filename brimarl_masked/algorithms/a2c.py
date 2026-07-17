import tensorflow as tf
import numpy as np
from brimarl_masked.agents.ac_agent_quick import ACAgentQuick
from brimarl_masked.algorithms.algorithm import Algorithm

class A2CAlgorithm(Algorithm):
    def __init__(self, num_players, discount, num_learning_per_epoch, min_samples=128, entropy_beta=0.0, epsilon=1e-8):
        self.num_players = num_players
        self.discount = discount
        # tf.optimizers.legacy was removed in Keras 3 (TF >= 2.16)
        self.optimizer_actor = tf.optimizers.Adam(1e-4)
        self.optimizer_critic = tf.optimizers.Adam(3e-4)
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

    def learn(self, agent: ACAgentQuick):
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

        s = tf.convert_to_tensor(self.s, dtype=tf.float32)
        a = tf.convert_to_tensor(self.a, dtype=tf.float32)
        r = tf.convert_to_tensor(self.r, dtype=tf.float32)
        sn = tf.convert_to_tensor(self.sn, dtype=tf.float32)
        d = tf.convert_to_tensor(self.d, dtype=tf.float32)
        m = tf.convert_to_tensor(self.m, dtype=tf.float32)

        loss = 0.
        iterations = 0.
        for _ in range(self.num_learning_per_epoch):
            # --- critic: V(s) towards the TD target r + γ (1 - d) V(s') ---
            with tf.GradientTape() as tape_critic:
                v_s = agent.value_net(s)
                v_sn = agent.value_net(sn)
                targets = tf.stop_gradient(r + self.discount * (1. - d) * v_sn)
                loss_critic = tf.reduce_mean(tf.square(targets - v_s))
            grads_critic = tape_critic.gradient(loss_critic, agent.value_net.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(grads_critic, agent.value_net.trainable_variables))

            # --- actor: policy gradient with advantage A = target - V(s) ---
            # the advantage uses the critic as fixed judge: no gradient through it
            advantages = tf.stop_gradient(targets - v_s)
            with tf.GradientTape() as tape_actor:
                probs = agent.policy_net(s)
                # renormalize over the legal actions, as done when acting
                masked_probs = probs * m
                masked_probs = masked_probs / (tf.reduce_sum(masked_probs, axis=1, keepdims=True) + self.epsilon)
                pi_a = tf.reduce_sum(masked_probs * a, axis=1, keepdims=True)
                log_pi_a = tf.math.log(pi_a + self.epsilon)
                entropy = -tf.reduce_sum(masked_probs * tf.math.log(masked_probs + self.epsilon), axis=1)
                loss_actor = -tf.reduce_mean(log_pi_a * advantages) - self.entropy_beta * tf.reduce_mean(entropy)
            grads_actor = tape_actor.gradient(loss_actor, agent.policy_net.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(grads_actor, agent.policy_net.trainable_variables))

            loss += loss_critic
            iterations += 1

        self.s = []
        self.a = []
        self.r = []
        self.sn = []
        self.d = []
        self.m = []
        return loss / (iterations + 1e-10) # in case something goes wrong and we divide by 0
