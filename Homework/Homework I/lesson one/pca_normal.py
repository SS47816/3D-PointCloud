# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud


def PCA(data: PyntCloud.points, correlation: bool=False, sort: bool=True) -> np.array:
    """ Calculate PCA

    Parameters
    ----------
        data(PyntCloud.points): 点云，NX3的矩阵
        correlation(bool): 区分np的cov和corrcoef，不输入时默认为False
        sort(bool): 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
    
    Returns
    ----------
        eigenvalues(np.array): 特征值
        eigenvectors(np.array): 特征向量

    """
    
    # 作业1
    # 屏蔽开始
    
    # Normalize X by the center
    X_ = data - np.mean(data, axis=0)
    # Compute SVD
    H = np.dot(X_.T, X_)
    # Compute Eigenvectors
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    ### Solution V1 ###
    # N = data.shape[0]
    # X = data.to_numpy()
    # # Normalization
    # X_ = X - np.mean(X, axis=0)
    # # Get function
    # func = np.cov if not correlation else np.corrcoef
    # H = func(X_, rowvar=False, bias=True)
    # # Compute eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = np.linalg.eig(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def get_pca_o3d(w: np.array, v: np.array, points: PyntCloud.points) -> o3d.geometry.LineSet:
    """ Build open3D geometry for PCA
    Parameters
    ----------
        w(np.array): eigenvalues in descending order
        v(np.array): eigenvectors in descending order
        points(np.array): pointcloud
    
    Returns
    ----------
        pca_set(o3d.geometry.LineSet): o3d line set for pca visualization
    """
    
    # calculate centroid & variation along main axis:
    centroid = points.mean()
    projs = np.dot(points.to_numpy(), v[:,0])
    scale = projs.max() - projs.min()

    points = centroid.to_numpy() + np.vstack(
        (
            np.asarray([0.0, 0.0, 0.0]),
            scale * v.T
        )
    ).tolist()
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    # from the largest to the smallest: RGB
    colors = np.identity(3).tolist()

    # build pca line set:
    pca_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pca_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pca_o3d

def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '../../../ModelNet40' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    # point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_pynt = PyntCloud.from_file("../../../ModelNet40/airplane/train/airplane_0001.ply")
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    N = points.shape[0]
    print('total number of points is:', N)

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    
    # TODO: 此处只显示了点云，还没有显示PCA
    pca_o3d = get_pca_o3d(w, v, points)
    # o3d.visualization.draw_geometries([point_cloud_o3d, pca_o3d])
    
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []

    # 作业2
    # 屏蔽开始
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    for i in range(N):
        [k, i, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 3)
        w, v = PCA(points.iloc[i])
        normals.append(v[:, 0])

    # 屏蔽结束

    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    points = np.vstack(
        (points.to_numpy(), points.to_numpy() + 5.0 * normals)
    )
    lines = [[i, i+N] for i in range(N)]
    colors = np.zeros((N, 3)).tolist()

    # build pca line set:
    surface_normals_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    surface_normals_o3d.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([point_cloud_o3d, pca_o3d, surface_normals_o3d])


if __name__ == '__main__':
    main()
