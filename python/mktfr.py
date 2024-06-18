import tensorflow as tf
import sys
from tqdm import tqdm


if __name__ == "__main__":
    yzx_file = sys.argv[1]
    tfr_file = sys.argv[2]

    with tf.io.TFRecordWriter(tfr_file) as tfd_writer:
        with open(yzx_file) as fin:
            for line in tqdm(fin):
                fields = line.rstrip().split(" ")
                clk = fields[0]
                feature_dict = {}
                for i in range(2, len(fields)):
                    slot, feat, _ = fields[i].split(":")
                    feature_dict.setdefault(slot, [])
                    feature_dict[slot].append(int(feat))

                tf_record = {}
                tf_record["click"] = tf.train.Feature(float_list=tf.train.FloatList(value=[int(clk)]))

                for slot, feat in feature_dict.items():
                    tf_record[slot] = tf.train.Feature(int64_list=tf.train.Int64List(value=feat))

                example = tf.train.Example(features=tf.train.Features(feature=tf_record))
                tfd_writer.write(example.SerializeToString())
    