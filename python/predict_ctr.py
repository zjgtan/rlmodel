from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import numpy as np
from lr import LR
from fm import FM
from dnn import DNN
import tensorflow as tf
from tqdm import tqdm
import sys
from feature_column import *

def parse_example(record):
    schema = {}
    schema["click"] = tf.io.FixedLenFeature((1,), tf.float32)
    for idx in range(27):
        schema[str(idx)] = tf.io.RaggedFeature(tf.int64)

    parsed_example = tf.io.parse_single_example(record, schema)

    for idx in range(27):
        parsed_example[str(idx)] = tf.ragged.stack([parsed_example[str(idx)]], axis=0)
    return parsed_example

def train_epoch(model, dataset, optimizer):
    for batch in tqdm(dataset):
        with tf.GradientTape() as tape:
            output_logits = model(batch["feat"])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["click"], logits=output_logits))
            click_auc = roc_auc_score(batch["click"], tf.nn.sigmoid(output_logits[0]))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(loss, click_auc)


def eval(model, dataset):
    click_logits_list, click_list = [], []

    for batch in tqdm(dataset):
        output_logits = model(batch["feat"])
        click_logits_list.append(output_logits)
        click_list.append(batch["click"])


    all_click_logits = tf.concat(click_logits_list, axis=0)
    all_click = tf.concat(click_list, axis=0)

    click_auc = roc_auc_score(all_click, tf.nn.sigmoid(all_click_logits))

    print("test: ", click_auc)

if __name__ == "__main__":
    camp = sys.argv[1]
    batch_size = 10000
    # 定义模型结构
    feature_map = dict((str(idx), VarLenColumn(str(idx), "general")) for idx in range(27))
    net = DNN(feature_map, 560871, 32, [256, 128])
    optimizer = tf.keras.optimizers.Adam()
    # 定义输入输出数据流
    train_set = tf.data.TFRecordDataset(["./data/{}/train.yzx.tfrecord".format(camp)]).map(lambda record: parse_example(record)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))
    test_set = tf.data.TFRecordDataset(["../data/{}/test.yzx.tfrecord".format(camp)]).map(lambda record: parse_example(record)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))

    train_epoch(net, train_set, optimizer)
    eval(net, test_set)  