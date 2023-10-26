# this file is to try to load the iris dataset for our proposed algorithms.

from sklearn.datasets import load_iris


# 加载Iris数据集
iris = load_iris()

# 现在，您可以使用iris变量访问数据集的特征和标签，例如：
X = iris.data  # 特征数据
y = iris.target  # 标签数据