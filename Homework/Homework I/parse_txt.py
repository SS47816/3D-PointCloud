import open3d as o3d
import numpy as np

def numpy_read_txt(pc_txt_path):
  fp = open(pc_txt_path,"r")
  all_lines = fp.readlines()
  data = np.genfromtxt(all_lines,delimiter=",")
  pc = data[:,:3]
  return pc

def parse_txt(pc_txt_path):
  fp = open(pc_txt_path,"r")
  all_lines = fp.readlines()
  data = []
  for line in all_lines:
    items = line.split(',')
    data.append([float(items[0]),float(items[1]),float(items[2])])
  
  data = np.array(data)
  return data

def visualize_pc(pc):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc)
  o3d.visualization.draw_geometries([pcd])

def main():
  pc_txt_path = "/Users/haoshuang/Documents/code/point_cloud_study/chapter1/Homework_I/pc/chair_0214.txt"
  #pc = numpy_read_txt(pc_txt_path)
  pc = parse_txt(pc_txt_path)
  visualize_pc(pc)

if __name__ == '__main__':
  main()