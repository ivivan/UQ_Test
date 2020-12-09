import tensorflow as tf
import numpy as np
import pandas as pd

def parse_tfrecord(example_proto):
    """The parsing function.

    Read a serialized example into the structure defined by featuresDict.

    Args:
        example_proto: a serialized Example.

    Returns:
        A tuple of the predictors dictionary and the label, cast to an `int32`.
    """
    parsed_features = tf.io.parse_single_example(example_proto, features_dict)
    labels = parsed_features.pop(LABEL)
    return parsed_features, tf.cast(labels, tf.int32,)


def to_tuple(inputs, label):
    return (list(inputs.values()),
            tf.squeeze(tf.one_hot(indices=label, depth=N_CLASSES)))




if __name__ == "__main__":

    # large
    TRAIN_FILE_PATH = "R:/CROPPHEN-Q2067/Data/DeepLearningTestData/MOREE/large_dataset/Training_moree_l.tfrecord.gz"
    TEST_FILE_PATH = "R:/CROPPHEN-Q2067/Data/DeepLearningTestData/MOREE/large_dataset/Testing_moree_l.tfrecord.gz"

    # small
    # TRAIN_FILE_PATH = "R:/CROPPHEN-Q2067/Data/DeepLearningTestData/MOREE/small_dataset/Training_moree_s.tfrecord.gz"
    # TEST_FILE_PATH = "R:/CROPPHEN-Q2067/Data/DeepLearningTestData/MOREE/small_dataset/Testing_moree_s.tfrecord.gz"

    logdir = "./crop_classification_multi/downloadtf"

    # Create a dataset from the TFRecord file in Cloud Storage.
    train_dataset = tf.data.TFRecordDataset(
        TEST_FILE_PATH, compression_type='GZIP')      # change here for train/test file path

    BANDS = ['00_EVI', '01_EVI', '02_EVI', '03_EVI', '04_EVI', '05_EVI', '06_EVI',
             '07_EVI', '08_EVI', '09_EVI', '10_EVI', '11_EVI', '12_EVI', '13_EVI',
             '14_EVI', '15_EVI', '16_EVI', '17_EVI', '18_EVI', '19_EVI', '20_EVI',
             '21_EVI', '22_EVI', '23_EVI', '24_EVI', '25_EVI', '26_EVI', '27_EVI',
             '28_EVI', '29_EVI', '30_EVI', '31_EVI', '32_EVI', '33_EVI', '34_EVI',
             '35_EVI', '36_EVI', '37_EVI', '38_EVI', '39_EVI','40_EVI']

    # BANDS = ['00_EVI', '01_EVI', '02_EVI', '03_EVI', '04_EVI', '05_EVI', '06_EVI',
    #         '07_EVI', '08_EVI', '09_EVI', '10_EVI', '11_EVI', '12_EVI', '13_EVI',
    #         '14_EVI', '15_EVI', '16_EVI', '17_EVI', '18_EVI', '19_EVI', '20_EVI']

    LABEL = 'class'
    N_CLASSES = 7
    FEATURE_NAMES = list(BANDS)
    FEATURE_NAMES.append(LABEL)

    # List of fixed-length features, all of which are float32.
    columns = [
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for k in FEATURE_NAMES
    ]

    # Dictionary with names as keys, features as values.

    features_dict = dict(zip(FEATURE_NAMES, columns))

    parsed_dataset = train_dataset.map(parse_tfrecord)

    input_dataset = parsed_dataset


    samples = []
    labels = []

    for element in input_dataset.as_numpy_iterator(): 
        temp = np.concatenate(list(element[0].values())).ravel()
        samples.append(temp)
        labels.append(element[1])

    X = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)


    df_x = pd.DataFrame(X)
    df_x.to_csv(f"{logdir}/test_x_large.csv", index=False)

    y = pd.DataFrame(y)
    y.to_csv(
        f"{logdir}/test_y_large.csv", index=False)



