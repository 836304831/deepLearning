
import os
import pickle
import numpy as np

__dir__ = os.path.abspath(os.path.dirname(__file__))

WORK_DIR = os.path.dirname(__dir__)
cifar10_path = os.path.join(WORK_DIR, "datasets/cifar-10-batches-py")


train_file_list = [os.path.join(cifar10_path, "data_batch_1"), os.path.join(cifar10_path, "data_batch_2"),
                   os.path.join(cifar10_path, "data_batch_3"), os.path.join(cifar10_path, "data_batch_4"),
                   os.path.join(cifar10_path, "data_batch_5")]

test_file_list = [os.path.join(cifar10_path, "test_batch")]


def load_data(file_list):
    """loaded data is dict which contain batch_label, labels, data, filenames
       batach_label: data description, type is str
       labels: the true label, value is 0 to 9, represent 10 category, type is int
       data: the train or predict feature data, is metric, type is np.ndarray
       filenames: the image name, an image represent a feature in data"""

    for file in file_list:
        with open(file, "rb") as f:
            loaded_data = pickle.load(f, encoding="ISO-8859-1")
            print("loaded_data type:", type(loaded_data))
            print("loaded_data keys:", loaded_data.keys())
            keys_list = list(loaded_data.keys())
            for key in keys_list:
                value = loaded_data.get(key)
                if type(value) == np.ndarray:
                    print("key name: {}, key type: {}, the first 3 data is:\n {}".format(key, type(value), value[0: 5]))
                    print("data.shape: {}".format(value.shape))
                else:
                    print("key name: {}, value type: {}, the data is \n {}".format(key, type(value), value))


if __name__ == "__main__":
    # load_data(test_file_list)
    load_data(train_file_list)
