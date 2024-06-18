from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
import numpy as np
from lr import LR
import tensorflow as tf
import tqdm



def parse_example(record, feature_map):
    schema = {}
    schema["feat"] = tf.io.FixedLenFeature((1, ), tf.int64)
    schema["click"] = tf.io.FixedLenFeature((1,), tf.float32)
    parsed_example = tf.io.parse_single_example(record, schema)
    return parsed_example

def train_epoch(model, dataset, optimizer):

    idx = 0
    for batch in tqdm(dataset):
        with tf.GradientTape() as tape:
            output_logits = model(batch)
            click_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["click"], logits=output_logits[0]))
            #conversion_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=batch["conversion"], logits=conversion_logits))
            try:
                click_auc = roc_auc_score(batch["click"], tf.nn.sigmoid(output_logits[0]))
                conversion_auc = 0
                #conversion_auc = roc_auc_score(batch["conversion"], tf.nn.sigmoid(conversion_logits))
            except:
                click_auc, conversion_auc = 0, 0

            #loss = click_loss + conversion_loss
            loss = click_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if idx % 100 == 0:
            print(loss, click_auc, conversion_auc)

        idx += 1

if __name__ == "__main__":
    # 定义模型结构
    net = LR(560871, 1)

    optimizer = tf.keras.optimizers.Adam()

    # 定义输入输出数据流
    alicpp_train_set = tf.data.TFRecordDataset(["../mtl/train.tfrecord"]).map(lambda record: parse_example(record, feature_map)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=yaml_config["batch_size"]))
    alicpp_test_set = tf.data.TFRecordDataset(["../mtl/test.tfrecord"]).map(lambda record: parse_example(record, feature_map)).apply(tf.data.experimental.dense_to_ragged_batch(batch_size=yaml_config["batch_size"]))

    for epoch in range(yaml_config["epoch"]):
        train_epoch(net, alicpp_train_set, optimizer)

    eval(net, alicpp_test_set)  