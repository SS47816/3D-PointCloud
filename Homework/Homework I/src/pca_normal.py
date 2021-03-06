# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import os
import time
import numpy as np
from pyntcloud import PyntCloud
import open3d as o3d


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
    # Get the H matrix (3x3)
    H = np.dot(X_.T, X_)
    # Compute SVD of H (Eigenvector of X = Eigenvector of H)
    # Get U, Sigma, V* (M = U Sigma V*)
    # V.columns are eigenvectors of M*M
    # U.columns are eigenvectors of MM*
    # Sigma.diagonal elements are non-negative roots of the eigenvalues of MM* and M*M
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def main():
    # 指定点云路径
    path = '../../../modelnet40_normal_resampled/'
    shape_name_list = np.loadtxt(os.path.join(path, 'modelnet40_shape_names.txt') if os.path.isdir(path) else None,dtype=str)

    for item in shape_name_list:
        # Import model
        filename = os.path.join(path, item, item+'_0001.txt')
        pointcloud = np.loadtxt(filename, delimiter=',')[:, 0:3]
        print('total points number is:', pointcloud.shape[0])
        
        # Convert to PyntCloud and Open3D formats
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud)
        # point_cloud_pynt = PyntCloud.from_instance("open3d", point_cloud_o3d)
        # points = point_cloud_pynt.points

        # 用PCA分析点云主方向
        N = pointcloud.shape[0]

        t0 = time.time()
        w, v = PCA(pointcloud)
        t1 = time.time()
        print('###### PCA time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))
        point_cloud_vector = v[:, 2] #点云主方向对应的向量
        print('the main orientation of this pointcloud is: ', point_cloud_vector)
        principle_axis = np.concatenate((np.array([[0.,0.,0.]]), v.T))
        print('Principal Axis: ', principle_axis)
        
        # Visualise the PCA Axis
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0,0,0))
        # Visualise the PCA Projection
        pr_data = pointcloud - np.dot(pointcloud, v[:,2][:,np.newaxis])*v[:, 2]
        pr_data = 1*v[:, 2] + pr_data
        
        pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
        pc_view.colors = o3d.utility.Vector3dVector([[0,0,0]])
        pr_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pr_data))
        # o3d.visualization.draw_geometries([pc_view, axis, pr_view])
        
        # 作业2
        # 屏蔽开始
        # 循环计算每个点的法向量
        # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
        # Feed the pointcloud into a kdtree structure
        t0 = time.time()
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        t1 = time.time()
        print('###### KDTreeFlann time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))
        normals = []
        
        t0 = time.time()
        for index in range(N):
            # For each point, search for its nearest k neighbors
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pc_view.points[index], 21)
            neighbor_pc = np.asarray(pc_view.points)[idx]
            # Compute the eigenvectors for its neighbors
            _, v = PCA(neighbor_pc)
            normals.append(v[:, 2])

        t1 = time.time()
        print('###### My Normal Estimation time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))

        t0 = time.time()
        point_cloud_o3d.estimate_normals()
        t1 = time.time()
        print('###### Open3D Normal Estimation time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))

        # 屏蔽结束

        # 此处把法向量存放在了normals中
        o3d_normals = np.asarray(point_cloud_o3d.normals, dtype=np.float64)
        normals = np.array(normals, dtype=np.float64)
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)

        # build pca line set:
        points = np.vstack((pointcloud, pointcloud + 0.03*normals))
        lines = [[i, i+N] for i in range(N)]
        colors = np.zeros((N, 3)).tolist()
        surface_normals_my = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        surface_normals_my.colors = o3d.utility.Vector3dVector(colors)

        points = np.vstack((pointcloud, pointcloud + 0.03*o3d_normals))
        lines = [[i, i+N] for i in range(N)]
        colors = np.full((N, 3), 0.5).tolist()
        surface_normals_o3d = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        surface_normals_o3d.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pc_view, axis, pr_view, surface_normals_my, surface_normals_o3d]) # point_cloud_o3d, 


if __name__ == '__main__':
    main()
