将NumPy的array视为矩阵时，你可以执行各种矩阵之间的常用操作。以下是一些常见的操作和使用NumPy的方法的总结：

    矩阵乘法：
        使用np.dot()函数或@运算符进行两个矩阵的矩阵乘法。

    python

result = np.dot(matrix1, matrix2)
# 或者
result = matrix1 @ matrix2

矩阵加法和减法：

    使用+和-运算符进行两个矩阵的加法和减法。

python

sum_matrix = matrix1 + matrix2
diff_matrix = matrix1 - matrix2

转置：

    使用.T属性或np.transpose()函数来获取矩阵的转置。

python

transposed_matrix = matrix.T
# 或者
transposed_matrix = np.transpose(matrix)

逆矩阵：

    使用np.linalg.inv()函数计算矩阵的逆矩阵。

python

inverse_matrix = np.linalg.inv(matrix)

行列式：

    使用np.linalg.det()函数计算矩阵的行列式。

python

determinant = np.linalg.det(matrix)

特征值和特征向量：

    使用np.linalg.eig()函数计算矩阵的特征值和特征向量。

python

eigenvalues, eigenvectors = np.linalg.eig(matrix)

迹(trace)：

    使用np.trace()函数计算矩阵的迹。

python

trace = np.trace(matrix)

矩阵的行数和列数：

    使用.shape属性获取矩阵的形状信息。

python

num_rows, num_cols = matrix.shape

矩阵的填充和切片：

    使用NumPy的切片和索引功能来访问和修改矩阵的元素。

python
这些是一些基本的矩阵操作，但NumPy还提供了更多高级的线性代数和数值计算功能，可以满足各种矩阵操作的需求。在实际应用中，你可以根据具体问题使用这些功能来操作和处理矩阵数据
    submatrix = matrix[1:3, 0:2]  # 获取子矩阵
    matrix[0, 0] = 42  # 修改矩阵元素




cp.multiply

@




