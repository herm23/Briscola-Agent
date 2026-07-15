import tensorflow as tf
import numpy as np
from brimarl_masked.agents.q_agent import DeepQAgent
from brimarl_masked.algorithms.algorithm import Algorithm
from brimarl_masked.algorithms.replay_memory import ReplayMemory


class QLearningAlgorithm(Algorithm):
    def __init__(self, num_players, batch_size, discount, replace_every,num_learning_per_epoch, replay_memory_capacity, loss_fn=None):
        self.num_players = num_players
        self.batch_size = batch_size
        self.discount = discount
        self.training_iterations = 0
        self.replace_every = replace_every
        self.target_net = None
        # tf.optimizers.legacy was removed in Keras 3 (TF >= 2.16)
        self.optimizer = tf.optimizers.Adam(1e-3)
        self.num_learning_per_epoch = num_learning_per_epoch
        self.replay_memory = ReplayMemory(replay_memory_capacity)
        # Interchangeable loss (e.g. tf.keras.losses.Huber(delta=...)); default MSE.
        self.loss_fn = loss_fn if loss_fn is not None else tf.keras.losses.MeanSquaredError()

    def store_game(self, states, actions, masks, rewards, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        masks = np.array(masks)
        dones = np.array(dones)

        next_states = np.copy(states)[self.num_players:]
        next_masks = np.copy(masks)[self.num_players:]
        dones = dones[self.num_players:]

        states = states[:-self.num_players]
        actions = actions[:-self.num_players]
        rewards = rewards[:-self.num_players]
        for i in range(0, len(states), self.num_players):
            self.replay_memory.push((
                states[i:i + self.num_players][tf.where(states[i:i + self.num_players][:, 0] == 1)[0]],
                next_states[i:i + self.num_players][tf.where(next_states[i:i + self.num_players][:, 0] == 1)[0]],
                actions[i],
                rewards[i],
                dones[i],
                masks[i],
                next_masks[i],
            ))

    def learn(self, agent: DeepQAgent):
        """Samples batch of experience from replay_memory,
        if it has enough samples, and trains the policy network on it
        """
        avg_loss = 0.
        if len(self.replay_memory.memory) >= self.batch_size:
            for _ in range(self.num_learning_per_epoch):
                if self.training_iterations % self.replace_every == 0:
                    print("resetting target net dqn")
                    self.target_net = agent.clone().policy_net

                self.training_iterations += 1

                # Otherwise sample a batch
                batch = self.replay_memory.sample(self.batch_size)

                states = []
                actions = []
                rewards = []
                done = []
                next_states = []
                masks = []
                next_masks = []
                for el in batch:
                    states.append(el[0])
                    next_states.append(el[1])
                    actions.append(el[2])
                    rewards.append(el[3])
                    done.append(el[4])
                    masks.append(el[5])
                    next_masks.append(el[6])

                states = tf.convert_to_tensor(np.array(states).squeeze(), dtype=tf.float32, )
                actions = tf.convert_to_tensor(np.array(actions), dtype=tf.float32, )
                rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32, )[..., None]
                done = tf.convert_to_tensor(np.array(done), dtype=tf.float32, )[..., None]
                next_states = tf.convert_to_tensor(np.array(next_states).squeeze(), dtype=tf.float32, )
                masks = tf.convert_to_tensor(np.array(masks), dtype=tf.float32, )
                next_masks = tf.convert_to_tensor(np.array(next_masks), dtype=tf.float32, )

                BS = states.shape[0]

                self.assert_same_shape(states, (BS, states.shape[1]))
                self.assert_same_shape(actions, (BS, actions.shape[1]))
                self.assert_same_shape(rewards, (BS, 1))
                self.assert_same_shape(next_states, states)
                self.assert_same_shape(done, (BS, 1))
                self.assert_same_shape(masks, masks)

                with tf.GradientTape() as tape:
                    # Q(s,a; θ): the net outputs 40 values, actions are one-hot
                    q_values = agent.policy_net(states)
                    q_sa = tf.reduce_sum(q_values * actions, axis=1, keepdims=True)

                    # TD target: y = r + γ (1 - done) max_a' Q(s', a'; θ⁻)
                    # computed with the frozen target net and restricted to the
                    # actions available in s' (next_masks), as required for convergence
                    next_q_values = self.target_net(next_states)
                    masked_next_q = next_q_values + (1. - next_masks) * np.finfo(np.float32).min
                    max_next_q = tf.reduce_max(masked_next_q, axis=1, keepdims=True)
                    targets = tf.stop_gradient(rewards + self.discount * (1. - done) * max_next_q)

                    q_learning_loss = self.loss_fn(targets, q_sa)

                gradients = tape.gradient(q_learning_loss, agent.policy_net.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, agent.policy_net.trainable_variables))

                avg_loss += float(q_learning_loss)

            agent.update_epsilon()
        return avg_loss / self.num_learning_per_epoch
