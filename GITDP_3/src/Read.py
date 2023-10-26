import numpy as np  # (activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)

from mlxtend.data import loadlocal_mnist
import json

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

def Read(par):
    ##### MNIST
    ##### type=np.ndarray (x_train: 60000 x 784 (float:0.~1.), y_train: 60000 x 1 (int:0~9), ... )
    ###########################################################################
    ##Read different dataset
    if par.Instance == "MNIST":
        x_train, y_train = loadlocal_mnist(images_path='./Inputs/train-images-idx3-ubyte',
                                           labels_path='./Inputs/train-labels-idx1-ubyte')
        x_test, y_test = loadlocal_mnist(images_path='./Inputs/t10k-images-idx3-ubyte',
                                         labels_path='./Inputs/t10k-labels-idx1-ubyte')

    elif par.Instance == "IRIS":
        iris = load_iris()
        xx = iris.data  # 特征数据
        yy = iris.target  # 标签数据

        x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=12)

    elif par.Instance == "WINE":
        wine = load_wine()
        xx = wine.data  # 特征数据
        yy = wine.target  # 标签数据

        x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=28, random_state=42)

    elif par.Instance == "TEST":
        seed_value = 1
        np.random.seed(seed_value)
        xx = 10*np.random.rand(3).reshape(-1, 1)
        yy = np.random.randint(0, 3, 3)

        x_train, x_test, y_train, y_test = xx, 0,yy,0

    elif par.Instance =="DIGITS":
        digits = load_digits()

        xx = digits.data  # 特征数据
        yy = digits.target  # 标签数据

        x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=297, random_state=42)

    elif par.Instance =="CANCER":
        cancer = load_breast_cancer()

        xx = cancer.data  # 特征数据
        yy = cancer.target  # 标签数据

        x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=59, random_state=par.randomseed)

# Load MNIST

    par.num_features = x_train.shape[1]  # 28*28
    unique_elements = set(y_train)
    count_unique_elements = len(unique_elements)
    par.num_classes = count_unique_elements  # 0 to 9 digits

    # Convert to float32.
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Flatten images to 1-D vector of 784 features (28*28).
    x_train, x_test = x_train.reshape([-1, par.num_features]), x_test.reshape([-1, par.num_features])

    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255.

    # Split data per agent
    x_train_agent = np.split(x_train, par.split_number)  ##  例如，如果 par.split_number 是 5，那么 x_train_agent 将会是一个包含 5 个子数组的列表，每个子数组都包含了 x_train 中的一部分数据
    y_train_agent = np.split(y_train, par.split_number)
    x_train_agent, y_train_agent = np.array(x_train_agent), np.array(y_train_agent)

    x_list = []
    y_list = []
    for p in range(par.split_number):
        x_list.append(x_train_agent[p])
        y_list.append(y_train_agent[p])

    x_train_new = np.concatenate(np.array(x_list))
    y_train_new = np.concatenate(np.array(y_list))

    return x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent ##似乎看来，新的数据集和旧的是一样的

