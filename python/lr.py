import sys
sys.path.append(".")
import tensorflow as tf
from tensorflow import keras

class LR(keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = self.create_embedding_layer(vocab_size)


    def create_embedding_layer(self, vocab_size):
        embedding_layer = keras.layers.Embedding(input_dim=vocab_size,
                                                    output_dim=self.embedding_dim)
        return embedding_layer

    def embedding_lookup(self, inputs):
        embedding = self.embedding_layer(inputs)
        return embedding

    def call(self, inputs):
        embedding = self.embedding_lookup(inputs)
        logits = tf.reduce_sum(embedding, axis=1)
        return logits

if __name__ == "__main__":
    model = LR(10, 1)
    print(model(tf.constant([[1,2], [3,4]])))
    