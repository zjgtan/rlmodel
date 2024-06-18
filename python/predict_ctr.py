from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import numpy as np
from lr import LR
import tensorflow as tf
import tqdm

def parse_example(record):
    schema = {}
    schema["feat"] = tf.io.FixedLenFeature((1, ), tf.int64)
    schema["click"] = tf.io.FixedLenFeature((1,), tf.float32)
    parsed_example = tf.io.parse_single_example(record, schema)
    return parsed_example

def train_epoch(model, dataset, optimizer):
    for batch in tqdm(dataset):
        with tf.GradientTape() as tape:
            output_logits = model(batch)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["click"], logits=output_logits))
            click_auc = roc_auc_score(batch["click"], tf.nn.sigmoid(output_logits[0]))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(loss, click_auc)

        idx += 1

if __name__ == "__main__":
    # 定义模型结构
    net = LR(560871, 1)

    optimizer = tf.keras.optimizers.Adam()

    # 定义输入输出数据流
    train_set = tf.data.TFRecordDataset(["../mtl/train.tfrecord"]).map(lambda record: parse_example(record)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=yaml_config["batch_size"]))
    test_set = tf.data.TFRecordDataset(["../mtl/test.tfrecord"]).map(lambda record: parse_example(record)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=yaml_config["batch_size"]))

    train_epoch(net, train_set, optimizer)
    eval(net, test_set)  