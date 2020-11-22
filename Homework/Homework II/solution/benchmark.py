# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree as octree
import kdtree as kdtree
from scipy.spatial import KDTree
from result_set import KNNResultSet, RadiusNNResultSet

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

#scipy knn查找
def scipy_kdtree_search(tree:KDTree,result_set:KNNResultSet,point: np.ndarray):
    scipy_dist,scipy_index=tree.query(point,result_set.capacity)
    for index, dist_index in enumerate(result_set.dist_index_list):
        dist_index.distance = scipy_dist[index]
        dist_index.index = scipy_index[index]
    return result_set

# brute force 查找
def brute_search(db: np.ndarray,result_set:KNNResultSet, query: np.ndarray):
    diff = np.linalg.norm(np.expand_dims(query, 0) - db, axis=1)
    nn_index = np.argsort(diff)
    nn_dist = diff[nn_index]
    for index, dist_index in enumerate(result_set.dist_index_list):
        dist_index.distance = nn_dist[index]
        dist_index.index = nn_index[index]
    return result_set

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    root_dir = './dataset/' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    print("octree --------------")
    construction_time_sum = 0
    scipy_construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    scipy_knn_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        #采样
        sample_index = np.random.choice(a=db_np.shape[0], size=30000, replace=False, p=None)
        db_np = db_np[sample_index,:]

        begin_t = time.time()
        root = octree.octree_construction(db_np, leaf_size, min_extent)
        construction_time_sum += time.time() - begin_t

        #scipy 建树
        begin_t = time.time()
        scipy_tree = KDTree(db_np)
        scipy_construction_time_sum += time.time() - begin_t

        for ii in range(db_np.shape[0]):
            query = db_np[ii,:]

            #octree 查找
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            octree.octree_knn_search(root, db_np, result_set, query)
            knn_time_sum += time.time() - begin_t

            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            octree.octree_radius_search_fast(root, db_np, result_set, query)
            radius_time_sum += time.time() - begin_t

            #scipy 查找
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            scipy_kdtree_search(scipy_tree,result_set,query)
            scipy_knn_time_sum += time.time() - begin_t

            #brute_force 查找
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            brute_search(db_np,result_set,query)
            brute_time_sum += time.time() - begin_t

    print("Octree: build %.3f, knn %.3f, radius %.3f, scipy %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     scipy_knn_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    scipy_construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    scipy_knn_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        #采样
        sample_index = np.random.choice(a=db_np.shape[0], size=30000, replace=False, p=None)
        db_np = db_np[sample_index,:]

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        #scipy 建树
        begin_t = time.time()
        scipy_tree = KDTree(db_np)
        scipy_construction_time_sum += time.time() - begin_t

        for jj in range(db_np.shape[0]):
            query = db_np[jj,:]

            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            kdtree.kdtree_knn_search(root, db_np, result_set, query)
            knn_time_sum += time.time() - begin_t

            begin_t = time.time()
            result_set = RadiusNNResultSet(radius=radius)
            kdtree.kdtree_radius_search(root, db_np, result_set, query)
            radius_time_sum += time.time() - begin_t

            #scipy 查找
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            scipy_kdtree_search(scipy_tree,result_set,query)
            scipy_knn_time_sum += time.time() - begin_t

            #brute_force 查找
            begin_t = time.time()
            result_set = KNNResultSet(capacity=k)
            brute_search(db_np,result_set,query)
            brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, scipy %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     scipy_knn_time_sum * 1000/iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))



if __name__ == '__main__':
    main()