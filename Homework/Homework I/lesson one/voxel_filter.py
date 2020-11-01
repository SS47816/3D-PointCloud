# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 
def voxel_filter(point_cloud, leaf_size, method='centroid'):
    """ 对点云进行voxel滤波
    
    Parameters
    ----------
        point_cloud：输入点云
        leaf_size: voxel尺寸

    Returns
    ----------
        filtered_points: filtered pointcloud

    """
    
    filtered_points = []
    # 作业3
    # 屏蔽开始
    
    # Define ROI
    (p_min, p_max) = (point_cloud.min(), point_cloud.max())
    (D_x, D_y, D_z) = (
        np.ceil((p_max['x'] - p_min['x']) / leaf_size).astype(np.int),
        np.ceil((p_max['y'] - p_min['y']) / leaf_size).astype(np.int),
        np.ceil((p_max['z'] - p_min['z']) / leaf_size).astype(np.int),
    )
    
    def classifier(x, y, z): 
        """ assign given 3D point to voxel grid
        
        Parameters
        ----------
            x(float): X
            y(float): Y
            z(float): Z

        Return
        ----------
            idx(int): voxel grid index
        """

        (i_x, i_y, i_z) = (
            np.floor((x - p_min['x']) / leaf_size).astype(np.int),
            np.floor((y - p_min['y']) / leaf_size).astype(np.int),            
            np.floor((z - p_min['z']) / leaf_size).astype(np.int),
        )

        idx = i_x + D_x * i_y + D_x * D_y * i_z

        return idx

    # assign to voxel grid:
    point_cloud['voxel_grid_id'] = point_cloud.apply(
        lambda row: classifier(row['x'], row['y'], row['z']), axis = 1
    )
    
    # centroid:
    if method == 'centroid':
        filtered_points = point_cloud.groupby(['voxel_grid_id']).mean().to_numpy()
    elif method == 'random':
        filtered_points = point_cloud.groupby(['voxel_grid_id']).apply(
            lambda x: x[['x', 'y', 'z']].sample(1)
        ).to_numpy()

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '../../../ModelNet40' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载自己的点云文件
    point_cloud_pynt = PyntCloud.from_file("../../../ModelNet40/airplane/train/airplane_0001.ply")

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 100.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
