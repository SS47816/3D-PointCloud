# 实现voxel滤波，并加载数据集中的文件进行验证
 
import os
import time
import numpy as np
import open3d as o3d
# from pyntcloud import PyntCloud

def voxel_filter(point_cloud: np.ndarray, leaf_size: float, method: str='centroid') -> np.array:
    """ 对点云进行voxel滤波
    
    Parameters
    ----------
        point_cloud(np.ndarray): 输入点云
        leaf_size(int): voxel尺寸
        method(str): select method 'centroid' or 'random'

    Returns
    ----------
        filtered_points(np.array): filtered pointcloud

    """
    
    filtered_points = []
    # 作业3
    # 屏蔽开始

    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

    Dx, Dy, Dz = (x_max - x_min)/leaf_size, (y_max - y_min)/leaf_size, (z_max - z_min)/leaf_size

    min_vec = np.array([x_min, y_min, z_min])
    indices = np.floor((point_cloud.copy() - min_vec)/leaf_size)
    index = indices[:, 0] + indices[:, 1]*Dx + indices[:, 2]*Dx*Dy

    for i in np.unique(index):
        voxel_points = point_cloud[index==i]
        if method == 'centroid':
            filtered_points.append(np.mean(voxel_points, axis=0))
        else:
            filtered_points.append(voxel_points[np.random.choice(a=voxel_points.shape[0])])
            
    # 屏蔽结束

    # Shift the coordintes of the filtered cloud (for visualization)
    filtered_points = np.add(filtered_points, [0., 1.2*Dy*leaf_size, 0.])
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # 指定点云路径
    path = '../../../modelnet40_normal_resampled/'
    shape_name_list = np.loadtxt(os.path.join(path, 'modelnet40_shape_names.txt') if os.path.isdir(path) else None,dtype=str)

    for item in shape_name_list:
        # Import model
        filename = os.path.join(path, item, item+'_0001.txt')
        pointcloud = np.loadtxt(filename, delimiter=',')[:, 0:3]
        print('total points number is:', pointcloud.shape[0])
        
        # Convert to Open3D formats
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(pointcloud)

        # 调用voxel滤波函数，实现滤波
        N = pointcloud.shape[0]

        t0 = time.time()
        filtered_cloud = voxel_filter(pointcloud, 0.08, 'centroid')
        t1 = time.time()
        print('###### My Voxel Downsample time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))
        print('Number of points in the filtered pointcloud:', filtered_cloud.shape[0])

        # Use Open3D functions to downsample the pointcloud
        t0 = time.time()
        filtered_cloud_o3d = point_cloud_o3d.voxel_down_sample(0.08)
        t1 = time.time()
        print('###### Open3D Voxel Downsample time taken (per 1k points): ', round((t1 - t0)/N*1000, 5))
        # Translate the filtered pointcloud
        translate_vector = -1.2*(filtered_cloud_o3d.get_max_bound() - filtered_cloud_o3d.get_min_bound())
        translate_vector[0] = 0.0
        translate_vector[2] = 0.0
        filtered_cloud_o3d.translate(translate_vector)

        # 显示滤波后的点云
        filtered_cloud_my = o3d.geometry.PointCloud()
        filtered_cloud_my.points = o3d.utility.Vector3dVector(filtered_cloud)
        o3d.visualization.draw_geometries([point_cloud_o3d, filtered_cloud_my, filtered_cloud_o3d])

if __name__ == '__main__':
    main()
