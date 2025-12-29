import os
import numpy as np
import math
import pandas
#import open3d as o3d
#from sklearn.neighbors import KDTree
#from scipy.spatial import ConvexHull
#import matplotlib.pyplot as plt
#from shapely.geometry import Point, Polygon
import cv2
import json

def calculate_angle_radians(x, y, cx, cy):
    # 计算起始点与圆心的连线与 x 轴的夹角（弧度）
    return math.atan2(y - cy, x - cx)

def calculate_euler_distance2d(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def calculate_euler_distance3d(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def calculate_point_on_arc(center_x, center_y, radius, start_angle, angle):
    # 计算弧上的点坐标
    x = center_x + radius * math.cos(start_angle + angle)
    y = center_y + radius * math.sin(start_angle + angle)
    return x, y


def rpy_to_rotation_matrix(rotation):
    theta = [rotation[2], rotation[0], rotation[1]] #roll pitch yaw (deg)
    theta = [i * math.pi/180 for i in theta]
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))


    return R


def extract3f_from_list(str_list):
    str_list = str_list[1:-1]  #get rid of '[]'
    ans_str = str_list.split(', ')
    ans = [float(ans_str[0]), float(ans_str[1]), float(ans_str[2])]
    return ans

def extract6f_from_list(str_list):
    str_list = str_list[1:-1]  #get rid of '[]'
    ans_str = str_list.split(', ')
    ans = [float(ans_str[0]), float(ans_str[1]), float(ans_str[2]), float(ans_str[3]), float(ans_str[4]), float(ans_str[5])]
    return ans


def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds

    Args:
        points (np.ndarray):
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.

    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """
    plane_list = []
    w_list = []
    N = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * N:
        if len(target) < 3:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target)

        w, index = pcd.segment_plane(threshold, 3, iterations)

        count += len(index)
        plane_list.append(np.asarray(target[index]))
        w_list.append(w)
        target = np.delete(target, index, axis=0)
    #print(len(target))
    return plane_list,w_list, target   #target is now outlier


def gen_new_coor(plane,w):
    #plane: a list of plane point clouds set
    min_x_point = min(plane, key=lambda point: point[0])
    origin = np.array(min_x_point)
    a,b,c,d = w
    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)  # get z axis
    x_axis_vector = None

    for point in plane:
        point_vector = np.array(point) - np.array(min_x_point)
        cross_product = np.cross(normal_vector, point_vector)
        if np.linalg.norm(cross_product) != 0:
            if np.dot(normal_vector, point_vector) < 0.01:
                x_axis_vector = point_vector
            break

    if x_axis_vector is None:
        raise ValueError("Cannot find a vector not collinear with the normal vector.")
    x_axis_vector /= np.linalg.norm(x_axis_vector)
    print('if the answer is 0',np.dot(normal_vector, x_axis_vector))
    y_axis_vector = np.cross(normal_vector, x_axis_vector)
    y_axis_vector /= np.linalg.norm(y_axis_vector)
    rotation_matrix = np.vstack((x_axis_vector, y_axis_vector, normal_vector))
    #print(rotation_matrix)
    # 构建变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    # transformation_matrix[:3, 3] = -origin.dot(rotation_matrix.T)
    transformation_matrix[:3, 3] = -np.dot(rotation_matrix, origin)
    #print('translation:',-origin.dot(rotation_matrix.T))
    #print('transformation matrix: ',transformation_matrix)
    #print(np.dot(transformation_matrix,np.array([origin[0],origin[1],origin[2],1])))

    return transformation_matrix

def trans_coor(plane,T):
    homogeneous_plane = np.hstack((plane, np.ones((len(plane), 1))))
    transformed_plane = np.dot(T, homogeneous_plane.T).T
    return transformed_plane[:,:3]

def spatial_partition(num_points, k, x_min, x_max, y_min, y_max):
    # Step 1: Calculate x and y ranges
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Step 2: Calculate total number of grids
    total_grids = num_points / k

    # Step 3: Calculate the square root of total number of grids
    sqrt_total_grids = np.sqrt(total_grids)

    # Step 4: Calculate number of grids in each direction
    num_grids_x = np.ceil(sqrt_total_grids * (x_range / max(x_range, y_range)))
    num_grids_y = np.ceil(sqrt_total_grids * (y_range / max(x_range, y_range)))

    # Step 5: Calculate width and height of each grid
    grid_size = max(x_range, y_range) / sqrt_total_grids

    # Step 6: Generate grid coordinates
    grid_coordinates = []
    centers = []
    for i in range(int(num_grids_x)):
        for j in range(int(num_grids_y)):
            center_x = x_min + (i + 0.5) * grid_size
            center_y = y_min + (j + 0.5) * grid_size

            # Adjust grid boundaries to cover the entire space
            min_x = center_x - 0.5 * grid_size
            max_x = center_x + 0.5 * grid_size
            min_y = center_y - 0.5 * grid_size
            max_y = center_y + 0.5 * grid_size

            # min_x = max(min_x, x_min)
            # max_x = min(max_x, x_max)
            # min_y = max(min_y, y_min)
            # max_y = min(max_y, y_max)

            grid_coordinates.append([min_x, max_x, min_y, max_y])
            centers.append([center_x, center_y])

    return grid_coordinates, centers


def points_in_grid(center_index, center, point_cloud, grid_coordinates):
    min_x, max_x, min_y, max_y = grid_coordinates[center_index]
    grid_center = center

    # 创建KD树
    kdtree = KDTree(point_cloud, leaf_size=30, metric='euclidean')

    # 查询KD树，找出距离网格中心在一定范围内的点
    indices = kdtree.query_radius([grid_center], r= (max_x - min_x) * math.sqrt(2))
    points_tmp = [point_cloud[i] for i in indices[0]]
    points = []
    for point in points_tmp:
        if point_belongs_to_grid(point, center_index, grid_coordinates):
            points.append(point)
    # 返回在网格内的点
    return points

def get_nearest_center(point, centers):
    # Initialize a KD tree with the center points
    kdtree = KDTree(centers)

    # Query KD tree to find the nearest center
    nearest_center_index = kdtree.query([point], k=1)[1][0]
    #nearest_center_index is a list
    return nearest_center_index

def point_belongs_to_grid(point, center_index, grid_coordinates):
    # Calculate grid boundaries
    min_x, max_x, min_y, max_y = grid_coordinates[center_index]
    # Check if the point is within the grid boundaries
    if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
        return True
    else:
        return False


def get_adjacent_grids(center_index, grid_coordinates, centers):
    grid_coordinates = np.array(grid_coordinates)
    current_center = centers[center_index]
    # print(current_center)
    grid_size = grid_coordinates[center_index,1] - grid_coordinates[center_index,0]
    # 计算周围网格的中心坐标
    left_center = (current_center[0] - grid_size, current_center[1])
    right_center = (current_center[0] + grid_size, current_center[1])
    up_center = (current_center[0], current_center[1] + grid_size)
    down_center = (current_center[0], current_center[1] - grid_size)

    up_left = (current_center[0] - grid_size, current_center[1] + grid_size)
    up_right = (current_center[0] + grid_size, current_center[1] + grid_size)
    down_left = (current_center[0] - grid_size, current_center[1] - grid_size)
    down_right = (current_center[0] + grid_size, current_center[1] - grid_size)
    # 初始化相邻网格的索引列表
    adjacent_grids = {'left': None, 'right': None, 'up': None, 'down': None}
    diag_grids = {'up_left': None, 'up_right': None, 'down_left': None, 'down_right': None}

    # 在中心坐标列表中查找相应的索引
    cnt_adj = 0
    for i, c in enumerate(centers):
        if calculate_euler_distance2d(c[0], c[1], left_center[0], left_center[1]) < 0.01:
            adjacent_grids['left'] = i
            cnt_adj += 1
        elif calculate_euler_distance2d(c[0], c[1], right_center[0], right_center[1]) < 0.01:
            adjacent_grids['right'] = i
            cnt_adj += 1
        elif calculate_euler_distance2d(c[0], c[1], up_center[0], up_center[1]) < 0.01:
            adjacent_grids['up'] = i
            cnt_adj += 1
        elif calculate_euler_distance2d(c[0], c[1], down_center[0], down_center[1]) < 0.01:
            adjacent_grids['down'] = i
            cnt_adj += 1
        elif calculate_euler_distance2d(c[0], c[1], up_left[0], up_left[1]) < 0.01:
            diag_grids['up_left'] = i
        elif calculate_euler_distance2d(c[0], c[1], up_right[0], up_right[1]) < 0.01:
            diag_grids['up_right'] = i
        elif calculate_euler_distance2d(c[0], c[1], down_left[0], down_left[1]) < 0.01:
            diag_grids['down_left'] = i
        elif calculate_euler_distance2d(c[0], c[1], down_right[0], down_right[1]) < 0.01:
            diag_grids['down_right'] = i

    return adjacent_grids, cnt_adj, diag_grids

def compute_largest_polygon(points):
    if len(points) < 3:
        print(points)
    hull = ConvexHull(points)
    points = np.asarray(points)
    largest_polygon = points[hull.vertices]
    return largest_polygon

# def is_point_inside_polygon(point, polygon):
#     x, y = point
#     inside = False
#     n = len(polygon)
#     p1x, p1y = polygon[0]
#     for i in range(n + 1):
#         p2x, p2y = polygon[i % n]
#         if y > min(p1y, p2y):
#             if y <= max(p1y, p2y):
#                 if x <= max(p1x, p2x):
#                     if p1y != p2y:
#                         xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                     if p1x == p2x or x <= xinters:
#                         inside = not inside
#         p1x, p1y = p2x, p2y
#     return inside

def is_point_in_polygon(x, y, polygon):
    point = Point(x, y)
    polygon_ = Polygon(polygon)
    return polygon_.contains(point)

def point_inside_boundary_1grid(own_grid_point, point_center, new_coordinates):
    #grid points, mean point of local pcd of object, test point
    if len(own_grid_point) == 0:
        return False
    elif len(own_grid_point) < 3:  # unable to form a polygon
        # then compare the distance from center point between test point and furthest grid point
        max_d = -1000
        for point in own_grid_point:
            d_tmp = calculate_euler_distance2d(point_center[0], point_center[1],
                                                     point[0], point[1])
            if d_tmp > max_d:
                max_d = d_tmp
        d_test_point = calculate_euler_distance2d(point_center[0], point_center[1],
                                                        new_coordinates[0], new_coordinates[1])
        if max_d >= d_test_point:
            return True
        else:
            return False
    else:
        polygon = compute_largest_polygon(own_grid_point)
        if is_point_in_polygon(new_coordinates[0], new_coordinates[1], polygon):
            return True
        else:
            return False

def find_voxel_index(point, voxel_width, num_voxel_x, num_voxel_y, num_voxel_z):
    x, y, z = point
    ix = int(np.floor(x / voxel_width))
    iy = int(np.floor((y + voxel_width * num_voxel_y / 2) / voxel_width))
    iz = int(np.floor((z + voxel_width * num_voxel_z / 2) / voxel_width))

    if 0 <= ix < num_voxel_x and 0 <= iy < num_voxel_y and 0 <= iz < num_voxel_z:
        index = iz + num_voxel_z * (iy + num_voxel_y * ix)
        return index
    else:
        return None  # point out of voxels

#-----------------------------------------------For VoD Dataset---------------------------------------------------------
def trans_point_coor(points,T):
    result = []
    for i in range(len(points)):
        point_1 = np.ones((4,1))
        point = points[i].reshape(3,1)
        point_1[0:3,:] = point
        tmp = T@point_1
        result.append(tmp[:3,:].reshape(1,3))
    return np.array(result).reshape(-1,3)

def get_intrinsic_matrix(txtfile):
    if os.path.exists(txtfile):
        file = open(txtfile)
        content = file.read()
        array = content.split('\n')
        array2 = array[2][4:].split(' ')
        array3 = np.array(array2, dtype=np.float64).reshape(-1, 4)
        intrinsic = array3[0:3,0:3]
        return intrinsic
    else:
        return None

def get_radar2cam(txtfile):
    if os.path.exists(txtfile):
        file = open(txtfile)
        content = file.read()
        array = content.split('\n')
        array2 = array[5][16:].split(' ')
        transformation = np.array(array2, dtype=np.float64).reshape(-1, 4)
        transformation_ = np.zeros((4,4))
        transformation_[3,3] = 1
        transformation_[0:3,:] = transformation
        return transformation_
    else:
        return None

def get_lidar2cam(txtfile):
    file = open(txtfile)
    content = file.read()
    array = content.split('\n')
    array2 = array[5][16:].split(' ')
    transformation = np.array(array2, dtype=np.float64).reshape(-1, 4)
    transformation_ = np.zeros((4,4))
    transformation_[3,3] = 1
    transformation_[0:3,:] = transformation
    return transformation_

def read_annotation(txtfile):
    annotations = []
    if os.path.exists(txtfile):
        with open(txtfile, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip():
                    data = line.split()
                    annotation = {
                        'Class': data[0],
                        'Track_ID': int(data[1]),
                        'Occluded': int(data[2]),
                        'Alpha': float(data[3]),
                        'Bbox': [float(coord) for coord in data[4:8]],
                        'Dimensions': [float(dim) for dim in data[8:11]],
                        'Location': [float(loc) for loc in data[11:14]],
                        'Rotation': float(data[14])
                    }
                    annotations.append(annotation)
        return annotations
    else:
        return None


def crop_image_around_point(image, x, y, offset):
    height, width = image.shape[:2]
    # 计算裁剪区域的左上角和右下角坐标
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(width, x + offset)
    y2 = min(height, y + offset)

    # 计算裁剪后的图像大小
    cropped_height = 2 * offset
    cropped_width = 2 * offset
    if x1 >= x2 or y1 >= y2:
        return np.zeros((cropped_height, cropped_width, image.shape[2]), dtype=image.dtype)
    # 创建一个空的图像用于裁剪，并填充为纯黑色
    cropped_image = np.zeros((cropped_height, cropped_width, image.shape[2]), dtype=image.dtype)

    # 计算在裁剪图像中的偏移量
    offset_x1 = max(0, offset - (x - x1))
    offset_y1 = max(0, offset - (y - y1))
    offset_x2 = min(2 * offset, offset_x1 + (x2 - x1))
    offset_y2 = min(2 * offset, offset_y1 + (y2 - y1))
    # 在裁剪图像中复制原始图像的部分
    cropped_image[offset_y1:offset_y2, offset_x1:offset_x2] = image[y1:y2, x1:x2]

    return cropped_image


def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    points_2d = []
    for point in points_3d:
        X, Y, Z = point
        x = (fx * X / Z) + cx
        y = (fy * Y / Z) + cy
        points_2d.append((x,y))
    return np.array(points_2d)

def filter_vlp16(points):
    new_pcl = []
    cnt = 0
    cnt_minus = 0
    for point in points:
        if point[3] == 0:
            cnt += 1
            continue
        elif point[0] < 0:
            cnt_minus += 1
        else:
            new_pcl.append(point)
    return np.array(new_pcl)


def visualize_with_image_color(image,points,color):

    for i in range(len(points)):
        cv2.circle(image,(int(points[i,0]),int(points[i,1])),3,color=color, thickness=-1)
    return image

def visulize_vod(pcd_radar,image,T_radar,intrinsic,color):
    geo_radar = pcd_radar[:,0:3]
    points_3d_r2c = trans_point_coor(geo_radar,T_radar)
    fx,cx = intrinsic[0,0],intrinsic[0,2]
    fy,cy = intrinsic[1,1],intrinsic[1,2]
    points_2d_radar = project_3d_to_2d(points_3d_r2c, fx, fy, cx, cy)
    new_image = visualize_with_image_color(image, points_2d_radar,color)
    cv2.imshow('new',new_image)
    cv2.waitKey(0)


def get_vod_dir(frame):
    base = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/'
    # base dir depended on your own settings.
    cam_base_dir = base + 'image_2/'
    radar_base_dir = base + 'velodyne/'
    calib_base_dir = base + 'calib/'
    txt_base_dir = '/workspace/data/VoD_dataset/label_2_with_track_ids/label_2/'
    lidar_base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/lidar/training/velodyne/'
    lidar_calib_base_dir = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/lidar/training/calib/'
    radar_file = radar_base_dir + str(frame).zfill(5) + '.bin'
    lidar_file = lidar_base_dir + str(frame).zfill(5) + '.bin'
    cam_file = cam_base_dir + str(frame).zfill(5) + '.jpg'
    calib_file = calib_base_dir + str(frame).zfill(5) + '.txt'
    lidar_calib_file = lidar_calib_base_dir + str(frame).zfill(5) + '.txt'
    return radar_file,lidar_file,cam_file,calib_file,lidar_calib_file,txt_base_dir


def crop_by_bbox(points, bbox_3d):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3d
    x_indices = np.logical_and(points[:, 0] >= xmin, points[:, 0] <= xmax)
    y_indices = np.logical_and(points[:, 1] >= ymin, points[:, 1] <= ymax)
    z_indices = np.logical_and(points[:, 2] >= zmin, points[:, 2] <= zmax)
    indices_within_bbox = np.logical_and.reduce((x_indices, y_indices, z_indices))
    points_within_bbox = points[indices_within_bbox]
    return points_within_bbox

def read_3dbbox(anno_1):
    x, y, z = anno_1['Location']
    l, w, h = anno_1['Dimensions']
    xmin, xmax = x - l / 2, x + l / 2
    ymin, ymax = y - w / 2, y + w / 2
    zmin, zmax = z - h / 2, z + h / 2
    bbox_3d = [xmin, xmax, ymin, ymax, zmin, zmax]
    loc = [x,y,z]
    return bbox_3d,loc
# def get_vod_readings(radar_file, lidar_file, cam_file, calib_file, lidar_calib_file):


def read_pose(frame_number):
    # 文件路径模板
    file_path_template = 'D:/VoD_dataset/view_of_delft_PUBLIC/radar_5frames/training/pose/{:05d}.json'

    # 根据帧数生成文件路径
    file_path = file_path_template.format(frame_number)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        return None, None, None
        #raise FileNotFoundError(f"File for frame {frame_number} does not exist at {file_path}")

    # 初始化三个坐标变换矩阵
    odom_to_camera = None
    map_to_camera = None
    utm_to_camera = None

    # 逐行读取文件内容并解析JSON对象
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'odomToCamera' in data:
                odom_to_camera = data['odomToCamera']
                odom_to_camera = np.array(odom_to_camera).reshape(4,4)
            elif 'mapToCamera' in data:
                map_to_camera = data['mapToCamera']
                map_to_camera = np.array(map_to_camera).reshape(4,4)
            elif 'UTMToCamera' in data:
                utm_to_camera = data['UTMToCamera']
                utm_to_camera = np.array(utm_to_camera).reshape(4,4)

    return odom_to_camera, map_to_camera, utm_to_camera


def compute_transform(frame0, frame1, T_radar0, T_radar1):
    odom_to_camera0, map_to_camera0, utm_to_camera0 = read_pose(frame0)
    odom_to_camera1, map_to_camera1, utm_to_camera1 = read_pose(frame1)
    if odom_to_camera0 is None or odom_to_camera1 is None:
        print('odom_to_camera not found.')
        return None, None, None
    else:
        odom_transform_radar, map_transform_radar, utm_transform_radar = None, None, None
    odom_transform_camera = np.dot(np.linalg.inv(odom_to_camera0), odom_to_camera1)
    T_radar0_inv = np.linalg.inv(T_radar0)
    odom_transform_radar = np.dot(np.dot(T_radar0_inv, odom_transform_camera), T_radar1)
    if map_to_camera0 is not None and utm_to_camera0 is not None and map_to_camera1 is not None and utm_to_camera1 is not None:
        map_transform_camera = np.dot(np.linalg.inv(map_to_camera0), map_to_camera1)
        utm_transform_camera = np.dot(np.linalg.inv(utm_to_camera0), utm_to_camera1)

        map_transform_radar = np.dot(np.dot(T_radar0_inv, map_transform_camera), T_radar1)
        utm_transform_radar = np.dot(np.dot(T_radar0_inv, utm_transform_camera), T_radar1)

    return odom_transform_radar, map_transform_radar, utm_transform_radar


def transform_bbox_to_radar(bbox, T_radar):
    T_camera_to_radar = np.linalg.inv(T_radar)

    corners = np.array([
        [bbox[0], bbox[2], bbox[4], 1],
        [bbox[0], bbox[2], bbox[5], 1],
        [bbox[0], bbox[3], bbox[4], 1],
        [bbox[0], bbox[3], bbox[5], 1],
        [bbox[1], bbox[2], bbox[4], 1],
        [bbox[1], bbox[2], bbox[5], 1],
        [bbox[1], bbox[3], bbox[4], 1],
        [bbox[1], bbox[3], bbox[5], 1]
    ])

    radar_corners = (np.dot(T_camera_to_radar, corners.T)).T

    x_coords = radar_corners[:, 0]
    y_coords = radar_corners[:, 1]
    z_coords = radar_corners[:, 2]

    radar_bbox = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max(), z_coords.min(), z_coords.max()]

    return radar_bbox

def transform_coor_to_radar(location, radar_transform_matrix):
    T_inv = np.linalg.inv(radar_transform_matrix)   #camera to radar
    location_homogeneous = np.array([location[0], location[1], location[2], 1])
    location_radar_homogeneous = np.dot(T_inv, location_homogeneous)
    location_radar = location_radar_homogeneous[:3]
    return location_radar


def filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw):
    """
    Filters points in a point cloud that are inside a 3D bounding box with rotation around the z-axis.

    Parameters:
    point_cloud (numpy array): The point cloud to filter (N x 4)
    bbox_location (array-like): The center of the bounding box [x, y, z]
    bbox_dimensions (array-like): The dimensions of the bounding box [length, width, height]
    bbox_yaw (float): The rotation of the bounding box around the z-axis in radians (clockwise)

    Returns:
    numpy array: The points inside the bounding box (M x 4)
    """
    # Extract bbox parameters
    bbox_center = np.array(bbox_location)
    bbox_height, bbox_length, bbox_width = bbox_dimensions
    yaw = bbox_yaw

    # Define rotation matrix for yaw (rotation around the z-axis)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Extract the 3D coordinates from the 4D point cloud
    points_xyz = point_cloud[:, :3]

    # Translate and rotate the point cloud to the bbox coordinate system
    translated_points = points_xyz - bbox_center
    local_points = np.dot(translated_points, Rz.T)

    # Check if the local points are within the bounds of the bbox
    mask = (
        (local_points[:, 0] >= -bbox_length / 2) & (local_points[:, 0] <= bbox_length / 2) &
        (local_points[:, 1] >= -bbox_width / 2) & (local_points[:, 1] <= bbox_width / 2) &
        (local_points[:, 2] >= -bbox_height / 2) & (local_points[:, 2] <= bbox_height / 2)
    )

    return point_cloud[mask], mask

def cartesian_to_spherical(v):
    v_x, v_y, v_z = v
    v_magnitude = np.sqrt(v_x**2 + v_y**2 + v_z**2)
    v_pitch = np.arctan2(v_z, np.sqrt(v_x ** 2 + v_y ** 2)) * 180 / np.pi  # 俯仰角
    v_yaw = np.arctan2(v_y, v_x) * 180 / np.pi  # 偏航角
    return v_magnitude, v_pitch, v_yaw


#-----------------------------------------------For VoD Dataset---------------------------------------------------------






#-----------------------------------------------For astyx Dataset---------------------------------------------------------
def get_asytx_dir(frame):
    base = 'D:/Astyx dataset/dataset_astyx_hires2019/'
    # base dir depended on your own settings.
    cam_base_dir = base + 'camera_front/'
    radar_base_dir = base + 'radar_6455/'
    calib_base_dir = base + 'calibration/'
    object_base_dir = base +'groundtruth_obj3d/'
    lidar_base_dir = base+'lidar_vlp16/'
    lidar_calib_base_dir = 'D:/VoD_dataset/view_of_delft_PUBLIC/lidar/training/calib/'
    radar_file = radar_base_dir + str(frame).zfill(6) + '.txt'
    lidar_file = lidar_base_dir + str(frame).zfill(6) + '.txt'
    cam_file = cam_base_dir + str(frame).zfill(6) + '.jpg'
    calib_file = calib_base_dir + str(frame).zfill(6) + '.json'
    object_file = object_base_dir + str(frame).zfill(6) + '.json'
    return radar_file,lidar_file,cam_file,calib_file,object_file


def get_transform_matrix(calib_file, sensor_uid_A, sensor_uid_B):
    #'radar_6455',"lidar_vlp16",'camera_front'
    with open(calib_file, 'r') as f:
        calibration_data = json.load(f)

    # 查找传感器A和传感器B的标定数据
    T_A_to_ref = None
    T_B_to_ref = None

    for sensor in calibration_data['sensors']:
        if sensor['sensor_uid'] == sensor_uid_A:
            T_A_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])
        elif sensor['sensor_uid'] == sensor_uid_B:
            T_B_to_ref = np.array(sensor['calib_data']['T_to_ref_COS'])

    # 检查是否找到了对应的传感器标定数据
    if T_A_to_ref is None or T_B_to_ref is None:
        return None

    # 计算从传感器A到传感器B的变换矩阵
    T_A_to_B = np.linalg.inv(T_B_to_ref) @ T_A_to_ref

    return T_A_to_B


def read_txt_data(file_path):
    #radar:x,y,z,v,r,mag
    #lidar:X Y Z Reflectivity LaserID Timestamp
    txt_data = np.genfromtxt(file_path, delimiter=' ', skip_header=1)
    txt_data = txt_data[~np.isnan(txt_data).any(axis=1)]

    return txt_data

#-----------------------------------------------For astyx Dataset--------------------------------------------------------



if __name__ == '__main__':
    print(rpy_to_rotation_matrix([-0.3,0.5,0.1]))