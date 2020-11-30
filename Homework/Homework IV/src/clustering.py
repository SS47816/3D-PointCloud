# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data: np.ndarray) -> np.ndarray :
    """
    Parameters
    ----------
    `data`: numpy.ndarray input pointcloud

    Returns
    ----------
    `segmengted_cloud`: numpy.ndarray

    """
    
    # 作业1
    # 屏蔽开始
    N, _ = data.shape
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    
    # keep points whose surface normal is approximate to z-axis for ground plane segementation:
    # pointcloud.estimate_normals(
    #     search_param = o3d.geometry.KDTreeSearchParamHybrid(
    #         radius = 0.50, max_nn = 9
    #     )
    # )
    # normals = np.asarray(pointcloud.normals)
    # angular_distance_to_z = np.abs(normals[:, 2])
    # filtered_idxs = angular_distance_to_z > np.cos(np.pi/6.0)
    # filtered_points = pointcloud[filtered_idxs]

    # Filter the ground plane using RANSAC
    plane, inlier_indices = pcd.segment_plane(
                                distance_threshold = 0.2, 
                                ransac_n = 5, 
                                num_iterations = 50
                                )
    t0 = time.time()
    pcd_filtered = pcd.select_by_index(inlier_indices, invert=True)
    # pcd_ground = pcd.select_by_index(inlier_indices, invert=False)
    t1 = time.time()
    print('###### Open3D Normal Estimation time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))
    
    segmengted_cloud = np.asarray(pcd_filtered.points())
    
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmengted_cloud.shape[0])
    return segmengted_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    

    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = '../../../kitti_point_clouds/' # 数据集路径
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
