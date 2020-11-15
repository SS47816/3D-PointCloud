# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.__K = n_clusters
        self.__max_iter = max_iter
        self.__priori = None
        self.__posteriori = None
        self.__mu = None
        self.__cov = None
    

    def get_mu(self):
        """
        Get mu
        """
        return np.copy(self.__mu)


    def __init_random(self, data):
        """
        Set initial GMM params with random initialization
        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        """
        N, _ = data.shape

        # init posteriori:
        self.__posteriori = np.zeros((self.__K, N))
        # init mu:
        self.__mu = data[np.random.choice(np.arange(N), size=self.__K, replace=False)]
        # init covariances
        self.__cov = np.asarray([np.cov(data, rowvar=False)] * self.__K)
        # init priori:
        self.__priori = np.ones((self.__K, 1)) / self.__K


    def __init_kmeans(self, data):
        """
        Set initial GMM params with K-Means initialization
        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        """
        N, _ = data.shape

        # init kmeans:
        k_means = KMeans(init='k-means++', n_clusters=self.__K)
        k_means.fit(data)
        category = k_means.labels_

        # init posteriori:
        self.__posteriori = np.zeros((self.__K, N))
        # init mu:
        self.__mu = k_means.cluster_centers_
        # init covariances
        self.__cov = np.asarray(
            [np.cov(data[category == k], rowvar=False) for k in range(self.__K)]
        )
        # init priori:
        value_counts = pd.Series(category).value_counts()
        self.__priori = np.asarray(
            [value_counts[k]/N for k in range(self.__K)]
        ).reshape((self.__K, 1))


    def __expectation_step(self, data):
        """
        Update posteriori
        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        """
        # expectation:
        for k in range(self.__K):
            self.__posteriori[k] = multivariate_normal.pdf(
                data, 
                mean=self.__mu[k], cov=self.__cov[k]
            )
        # get posteriori:
        self.__posteriori = np.dot(
            np.diag(self.__priori.ravel()), self.__posteriori
        )
        # normalize:
        self.__posteriori /= np.sum(self.__posteriori, axis=0)

    def __maximization_step(self, data, N_k, N):
        """
        Update posteriori
        Parameters
        ----------
        data: numpy.ndarray
            Training set as N-by-D numpy.ndarray
        """
        self.mu = np.asarray(
            [np.dot(self.__posteriori[k], data)/N_k[k] for k in range(self.__K)]
        )
        self.__cov = np.asarray(
            [
                np.dot(
                    (data - self.__mu[k]).T, 
                    np.dot(np.diag(self.__posteriori[k].ravel()), data - self.__mu[k])
                )/N_k[k] for k in range(self.__K)
            ]  
        )
        self.__priori = (N_k/N).reshape((self.__K, 1))

    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        
        N, _ = data.shape
        self.__init_kmeans(data)
        
        for i in range(self.__max_iter):
            # Expectation
            self.__expectation_step(data)
            # Count the number of points in the cluster
            N_k = np.sum(self.__posteriori, axis=1)
            # Maximization
            self.__maximization_step(data, N_k, N)

        # 屏蔽结束
    

    def predict(self, data):
        """
        Classify input data

        Parameters
        ----------
        data: numpy.ndarray
            Testing set as N-by-D numpy.ndarray

        Returns
        ----------
        result: numpy.ndarray
            data labels as (N, ) numpy.ndarray
        """
        # 屏蔽开始

        self.__expectation_step(data)

        result = np.argmax(self.__posteriori, axis = 0)

        return result

        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

