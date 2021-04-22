import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def dist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# 随机找k个点作为初始点
def center_choose(dataset, k):
    m, n = dataset.shape
    center = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))
        center[i] = np.array(dataset[index])
    return center

# k-means 均值聚类
def Kmeans(dataset, k):
    row_num = np.shape(dataset)[0]
    center_loss = np.mat(np.zeros((row_num, 2)))
    print(center_loss)
    center_change_flag = True
    # 初始化中心
    center = center_choose(dataset, k)
    count = 1
    while center_change_flag:
        center_change_flag = False
        # 遍历所有样本
        for i in range(row_num):
            min_distance = 100
            min_index = -1
            # 找出最近的中心
            for j in range(k):
                distance = dist(center[j, :], np.array(dataset[i, :]))
                if distance < min_distance:
                    min_index = j
                    min_distance = distance
            # 更新中心
            if center_loss[i, 0] != min_index:
                center_change_flag = True
                center_loss[i, :] = min_index, min_distance**2
        for j in range(k):
            cla = np.array(dataset[np.nonzero(center_loss[:, 0].A == j)[0]])
            center[j, :] = np.mean(cla, axis=0)
        print("第 ", count, "次聚类完成")
        count = count + 1
    return center, center_loss
#                                     数据读取与预处理
path = 'NBA_Season_Stats(1984-2015)-new.csv'
data = pd.read_csv(path, encoding='gbk')
data = data.dropna()
data = data.reset_index(drop=True)
# 去掉没用属性
X = data.drop(['Pos'], 1)
X1 = data
# Max-Min 归一化
# X = (X - X.min()) / (X.max() - X.min())
# Z-score归一化
X = (X - X.mean()) / (X.std())
# 降维
# pca = PCA(n_components=2)   # 降维
# pca.fit(X)                  # 训练
# X = pca.fit_transform(X)
# print(X)
print(type(X))

k = 5
center, loss = Kmeans(X, k)
print(center)


colors = ['black', 'blue', 'darkred', 'hotpink', 'green']
fig = plt.figure()
for j in range(5):
    cla = np.array(X[np.nonzero(loss[:, 0].A == j)[0]])
    plt.scatter(np.array(cla[:, 0]), np.array(cla[:, 1]), c=colors[j])
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(np.array(X[:, 0]), np.array(X[:, 1]))
plt.show()
