import tensorflow as tf
from tensorflow import Tensor


class DQN(tf.keras.models.Model):
    def __init__(self,
                 num_actions: int,
                 hidden_size: int = 256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Q-network: state (244) -> Q(s, a) for each of the 40 cards.
        # Linear output layer: Q-values are expected returns, they can be negative.
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(num_actions),
        ])

    def build(self, input_shape):
        self.model.build(input_shape)
        self.built = True

    def call(self, x: Tensor):
        return self.model(x)
