import tensorflow as tf
from tensorflow import Tensor


class Actor(tf.keras.models.Model):
    def __init__(self,
                 num_actions: int = 40,
                 hidden_size: int = 256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Policy network: state (284) -> pi(a|s) for each of the 40 cards.
        # Softmax output: ACAgentQuick.action expects probabilities, which it
        # masks and renormalizes over the cards actually in hand.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(num_actions, activation="softmax"),
        ])

    def build(self, input_shape):
        self.model.build(input_shape)
        self.built = True

    def call(self, x: Tensor):
        return self.model(x)


class Critic(tf.keras.models.Model):
    def __init__(self,
                 hidden_size: int = 256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Value network: state (284) -> V(s), a single scalar.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def build(self, input_shape):
        self.model.build(input_shape)
        self.built = True

    def call(self, x: Tensor):
        return self.model(x)
