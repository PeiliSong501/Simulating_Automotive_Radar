import open3d as o3d
import numpy as np

# 创建两个示例点云
def create_point_cloud(points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

# 示例点云A
points_A = np.array([
    [0.5, 2.3, 3.1],
    [0.1, 0.4, 2.2],
    [0.3, 1.2, 0.5],
    [0.7, 3.4, 1.8]
])

# 示例点云B
points_B = np.array([
    [0.6, 2.5, 3.0],
    [0.2, 0.5, 2.3],
    [0.4, 1.3, 0.6],
    [0.8, 3.5, 1.9]
])

source = create_point_cloud(points_A)
target = create_point_cloud(points_B)

# 执行ICP配准
threshold = 0.5  # 设置最近邻点的最大距离
trans_init = np.eye(4)  # 初始化变换矩阵

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)
fitness = reg_p2p.fitness
rmse = reg_p2p.inlier_rmse

print(f"Fitness: {fitness}")
print(f"RMSE: {rmse}")
# 打印结果
print("Transformation matrix:")
print(reg_p2p.transformation)

# 将源点云转换到目标点云的坐标系下
transformed_source = source.transform(reg_p2p.transformation)

# 可视化配准后的点云
o3d.visualization.draw_geometries([transformed_source, target])

# 获取转换后的点云数组
transformed_points_A = np.asarray(transformed_source.points)

# 打印转换后的点云数组
print("Transformed Source Points:")
print(transformed_points_A)