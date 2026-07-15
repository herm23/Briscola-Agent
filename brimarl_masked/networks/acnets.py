import tensorflow as tf
from tensorflow import Tensor


class Actor(tf.keras.models.Model):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras...
        ])
    def call(self, x: Tensor):
        return self.model(x)

class Critic(tf.keras.models.Model):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model = tf.keras...
    def call(self, x: Tensor):
        return self.model(x)

