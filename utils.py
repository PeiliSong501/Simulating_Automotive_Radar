import os
import json
import math
import cv2
import numpy as np
import pandas

# Optional dependencies.
# Uncomment them if you need the corresponding functionality.
# import open3d as o3d
# from sklearn.neighbors import KDTree
# from scipy.spatial import ConvexHull
# from shapely.geometry import Point, Polygon
# import matplotlib.pyplot as plt


# ============================================================
# Reproducible path configuration
# Modify only the paths below when reproducing this project.
# ============================================================
VOD_ROOT = "/workspace/data/VoD_dataset/view_of_delft_PUBLIC"
VOD_RADAR_TRAIN_ROOT = os.path.join(VOD_ROOT, "radar", "training")
VOD_LIDAR_TRAIN_ROOT = os.path.join(VOD_ROOT, "lidar", "training")
VOD_LABEL_ROOT = "/workspace/data/VoD_dataset/label_2_with_track_ids/label_2"
VOD_POSE_TEMPLATE = os.path.join(
    VOD_ROOT, "radar_5frames", "training", "pose", "{:05d}.json"
)

ASTYX_ROOT = "D:/Astyx dataset/dataset_astyx_hires2019"


"""
What this file does
-------------------
This utility file provides helper functions used throughout the project.

Main functionality includes:
1. Geometry and distance computation.
2. Coordinate transformations between sensor frames.
3. Camera projection and visualization helpers.
4. Dataset path generation for VoD and Astyx.
5. Annotation parsing and 3D bounding box processing.
6. Pose loading and frame-to-frame motion transformation.

In short, this file serves as the shared toolbox for data loading,
coordinate conversion, projection, visualization, and bounding box utilities.
"""


def calculate_angle_radians(x, y, cx, cy):
    """Compute the angle (in radians) between the point-center vector and the x-axis."""
    return math.atan2(y - cy, x - cx)


def calculate_euler_distance2d(x1, y1, x2, y2):
    """Compute the Euclidean distance between two 2D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_euler_distance3d(x1, y1, z1, x2, y2, z2):
    """Compute the Euclidean distance between two 3D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def calculate_point_on_arc(center_x, center_y, radius, start_angle, angle):
    """Compute a point on an arc given the center, radius, and angles."""
    x = center_x + radius * math.cos(start_angle + angle)
    y = center_y + radius * math.sin(start_angle + angle)
    return x, y


def rpy_to_rotation_matrix(rotation):
    """Convert roll-pitch-yaw angles (in degrees) to a rotation matrix."""
    theta = [rotation[2], rotation[0], rotation[1]]
    theta = [angle * math.pi / 180 for angle in theta]

    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta[0]), -math.sin(theta[0])],
        [0, math.sin(theta[0]), math.cos(theta[0])],
    ])

    rotation_y = np.array([
        [math.cos(theta[1]), 0, math.sin(theta[1])],
        [0, 1, 0],
        [-math.sin(theta[1]), 0, math.cos(theta[1])],
    ])

    rotation_z = np.array([
        [math.cos(theta[2]), -math.sin(theta[2]), 0],
        [math.sin(theta[2]), math.cos(theta[2]), 0],
        [0, 0, 1],
    ])

    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix


def extract3f_from_list(str_list):
    """Parse a string list like '[a, b, c]' into three floats."""
    str_list = str_list[1:-1]
    values = str_list.split(", ")
    return [float(values[0]), float(values[1]), float(values[2])]


def extract6f_from_list(str_list):
    """Parse a string list like '[a, b, c, d, e, f]' into six floats."""
    str_list = str_list[1:-1]
    values = str_list.split(", ")
    return [
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    ]


def DetectMultiPlanes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """Detect multiple planes from a point cloud using iterative plane segmentation.

    Note:
        This function requires Open3D. Uncomment the Open3D import at the top
        of this file before using it.
    """
    plane_list = []
    plane_equation_list = []
    total_num_points = len(points)
    remaining_points = points.copy()
    count = 0

    while count < (1 - min_ratio) * total_num_points:
        if len(remaining_points) < 3:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining_points)

        plane_equation, inlier_indices = pcd.segment_plane(threshold, 3, iterations)
        count += len(inlier_indices)
        plane_list.append(np.asarray(remaining_points[inlier_indices]))
        plane_equation_list.append(plane_equation)
        remaining_points = np.delete(remaining_points, inlier_indices, axis=0)

    return plane_list, plane_equation_list, remaining_points


def gen_new_coor(plane, plane_equation):
    """Generate a local coordinate system from a plane point set and plane equation."""
    min_x_point = min(plane, key=lambda point: point[0])
    origin = np.array(min_x_point)

    a, b, c, d = plane_equation
    normal_vector = np.array([a, b, c])
    normal_vector /= np.linalg.norm(normal_vector)

    x_axis_vector = None
    for point in plane:
        point_vector = np.array(point) - np.array(min_x_point)
        cross_product = np.cross(normal_vector, point_vector)
        if np.linalg.norm(cross_product) != 0:
            if np.dot(normal_vector, point_vector) < 0.01:
                x_axis_vector = point_vector
            break

    if x_axis_vector is None:
        raise ValueError("Cannot find a vector that is not collinear with the normal vector.")

    x_axis_vector /= np.linalg.norm(x_axis_vector)
    print("Orthogonality check (should be close to 0):", np.dot(normal_vector, x_axis_vector))

    y_axis_vector = np.cross(normal_vector, x_axis_vector)
    y_axis_vector /= np.linalg.norm(y_axis_vector)

    rotation_matrix = np.vstack((x_axis_vector, y_axis_vector, normal_vector))

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = -np.dot(rotation_matrix, origin)

    return transformation_matrix


def trans_coor(plane, transformation_matrix):
    """Apply a homogeneous transformation matrix to a point set."""
    homogeneous_plane = np.hstack((plane, np.ones((len(plane), 1))))
    transformed_plane = np.dot(transformation_matrix, homogeneous_plane.T).T
    return transformed_plane[:, :3]


def spatial_partition(num_points, k, x_min, x_max, y_min, y_max):
    """Partition a 2D space into grids based on the expected number of points per cell."""
    x_range = x_max - x_min
    y_range = y_max - y_min
    total_grids = num_points / k
    sqrt_total_grids = np.sqrt(total_grids)

    num_grids_x = np.ceil(sqrt_total_grids * (x_range / max(x_range, y_range)))
    num_grids_y = np.ceil(sqrt_total_grids * (y_range / max(x_range, y_range)))
    grid_size = max(x_range, y_range) / sqrt_total_grids

    grid_coordinates = []
    centers = []
    for i in range(int(num_grids_x)):
        for j in range(int(num_grids_y)):
            center_x = x_min + (i + 0.5) * grid_size
            center_y = y_min + (j + 0.5) * grid_size

            min_x = center_x - 0.5 * grid_size
            max_x = center_x + 0.5 * grid_size
            min_y = center_y - 0.5 * grid_size
            max_y = center_y + 0.5 * grid_size

            grid_coordinates.append([min_x, max_x, min_y, max_y])
            centers.append([center_x, center_y])

    return grid_coordinates, centers


def points_in_grid(center_index, center, point_cloud, grid_coordinates):
    """Return all points that belong to a specific grid cell.

    Note:
        This function requires sklearn.neighbors.KDTree. Uncomment the KDTree
        import at the top of this file before using it.
    """
    min_x, max_x, min_y, max_y = grid_coordinates[center_index]
    grid_center = center

    kdtree = KDTree(point_cloud, leaf_size=30, metric="euclidean")
    indices = kdtree.query_radius([grid_center], r=(max_x - min_x) * math.sqrt(2))

    points_tmp = [point_cloud[i] for i in indices[0]]
    points = []
    for point in points_tmp:
        if point_belongs_to_grid(point, center_index, grid_coordinates):
            points.append(point)
    return points


def get_nearest_center(point, centers):
    """Find the nearest grid center to a point.

    Note:
        This function requires sklearn.neighbors.KDTree.
    """
    kdtree = KDTree(centers)
    nearest_center_index = kdtree.query([point], k=1)[1][0]
    return nearest_center_index


def point_belongs_to_grid(point, center_index, grid_coordinates):
    """Check whether a point belongs to a specific 2D grid cell."""
    min_x, max_x, min_y, max_y = grid_coordinates[center_index]
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def get_adjacent_grids(center_index, grid_coordinates, centers):
    """Find the indices of adjacent and diagonal neighboring grids."""
    grid_coordinates = np.array(grid_coordinates)
    current_center = centers[center_index]
    grid_size = grid_coordinates[center_index, 1] - grid_coordinates[center_index, 0]

    left_center = (current_center[0] - grid_size, current_center[1])
    right_center = (current_center[0] + grid_size, current_center[1])
    up_center = (current_center[0], current_center[1] + grid_size)
    down_center = (current_center[0], current_center[1] - grid_size)

    up_left = (current_center[0] - grid_size, current_center[1] + grid_size)
    up_right = (current_center[0] + grid_size, current_center[1] + grid_size)
    down_left = (current_center[0] - grid_size, current_center[1] - grid_size)
    down_right = (current_center[0] + grid_size, current_center[1] - grid_size)

    adjacent_grids = {"left": None, "right": None, "up": None, "down": None}
    diagonal_grids = {
        "up_left": None,
        "up_right": None,
        "down_left": None,
        "down_right": None,
    }

    count_adjacent = 0
    for i, center in enumerate(centers):
        if calculate_euler_distance2d(center[0], center[1], left_center[0], left_center[1]) < 0.01:
            adjacent_grids["left"] = i
            count_adjacent += 1
        elif calculate_euler_distance2d(center[0], center[1], right_center[0], right_center[1]) < 0.01:
            adjacent_grids["right"] = i
            count_adjacent += 1
        elif calculate_euler_distance2d(center[0], center[1], up_center[0], up_center[1]) < 0.01:
            adjacent_grids["up"] = i
            count_adjacent += 1
        elif calculate_euler_distance2d(center[0], center[1], down_center[0], down_center[1]) < 0.01:
            adjacent_grids["down"] = i
            count_adjacent += 1
        elif calculate_euler_distance2d(center[0], center[1], up_left[0], up_left[1]) < 0.01:
            diagonal_grids["up_left"] = i
        elif calculate_euler_distance2d(center[0], center[1], up_right[0], up_right[1]) < 0.01:
            diagonal_grids["up_right"] = i
        elif calculate_euler_distance2d(center[0], center[1], down_left[0], down_left[1]) < 0.01:
            diagonal_grids["down_left"] = i
        elif calculate_euler_distance2d(center[0], center[1], down_right[0], down_right[1]) < 0.01:
            diagonal_grids["down_right"] = i

    return adjacent_grids, count_adjacent, diagonal_grids


def compute_largest_polygon(points):
    """Compute the convex hull polygon for a 2D point set.

    Note:
        This function requires scipy.spatial.ConvexHull.
    """
    if len(points) < 3:
        print(points)
    hull = ConvexHull(points)
    points = np.asarray(points)
    largest_polygon = points[hull.vertices]
    return largest_polygon


def is_point_in_polygon(x, y, polygon):
    """Check whether a point is inside a polygon.

    Note:
        This function requires shapely.geometry.Point and Polygon.
    """
    point = Point(x, y)
    polygon_ = Polygon(polygon)
    return polygon_.contains(point)


def point_inside_boundary_1grid(own_grid_point, point_center, new_coordinates):
    """Check whether a point lies inside the local boundary formed by points in one grid."""
    if len(own_grid_point) == 0:
        return False
    elif len(own_grid_point) < 3:
        # If a polygon cannot be formed, compare distances to the center.
        max_distance = -1000
        for point in own_grid_point:
            distance_tmp = calculate_euler_distance2d(
                point_center[0], point_center[1], point[0], point[1]
            )
            if distance_tmp > max_distance:
                max_distance = distance_tmp

        test_distance = calculate_euler_distance2d(
            point_center[0], point_center[1], new_coordinates[0], new_coordinates[1]
        )
        return max_distance >= test_distance
    else:
        polygon = compute_largest_polygon(own_grid_point)
        return is_point_in_polygon(new_coordinates[0], new_coordinates[1], polygon)


def find_voxel_index(point, voxel_width, num_voxel_x, num_voxel_y, num_voxel_z):
    """Map a point to its flattened voxel index."""
    x, y, z = point
    ix = int(np.floor(x / voxel_width))
    iy = int(np.floor((y + voxel_width * num_voxel_y / 2) / voxel_width))
    iz = int(np.floor((z + voxel_width * num_voxel_z / 2) / voxel_width))

    if 0 <= ix < num_voxel_x and 0 <= iy < num_voxel_y and 0 <= iz < num_voxel_z:
        return iz + num_voxel_z * (iy + num_voxel_y * ix)
    return None


# ============================================================
# Utilities for the VoD dataset
# ============================================================
def trans_point_coor(points, transformation_matrix):
    """Transform 3D points with a 4x4 homogeneous transformation matrix."""
    result = []
    for i in range(len(points)):
        point_homogeneous = np.ones((4, 1))
        point = points[i].reshape(3, 1)
        point_homogeneous[0:3, :] = point
        transformed = transformation_matrix @ point_homogeneous
        result.append(transformed[:3, :].reshape(1, 3))
    return np.array(result).reshape(-1, 3)


def get_intrinsic_matrix(txtfile):
    """Read the camera intrinsic matrix from a VoD calibration file."""
    if os.path.exists(txtfile):
        with open(txtfile) as file:
            content = file.read()
        array = content.split("\n")
        array2 = array[2][4:].split(" ")
        array3 = np.array(array2, dtype=np.float64).reshape(-1, 4)
        intrinsic = array3[0:3, 0:3]
        return intrinsic
    return None


def get_radar2cam(txtfile):
    """Read the radar-to-camera transformation matrix from a VoD calibration file."""
    if os.path.exists(txtfile):
        with open(txtfile) as file:
            content = file.read()
        array = content.split("\n")
        array2 = array[5][16:].split(" ")
        transformation = np.array(array2, dtype=np.float64).reshape(-1, 4)
        transformation_4x4 = np.zeros((4, 4))
        transformation_4x4[3, 3] = 1
        transformation_4x4[0:3, :] = transformation
        return transformation_4x4
    return None


def get_lidar2cam(txtfile):
    """Read the LiDAR-to-camera transformation matrix from a VoD calibration file."""
    with open(txtfile) as file:
        content = file.read()
    array = content.split("\n")
    array2 = array[5][16:].split(" ")
    transformation = np.array(array2, dtype=np.float64).reshape(-1, 4)
    transformation_4x4 = np.zeros((4, 4))
    transformation_4x4[3, 3] = 1
    transformation_4x4[0:3, :] = transformation
    return transformation_4x4


def read_annotation(txtfile):
    """Read object annotations from a VoD label file."""
    annotations = []
    if os.path.exists(txtfile):
        with open(txtfile, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip():
                    data = line.split()
                    annotation = {
                        "Class": data[0],
                        "Track_ID": int(data[1]),
                        "Occluded": int(data[2]),
                        "Alpha": float(data[3]),
                        "Bbox": [float(coord) for coord in data[4:8]],
                        "Dimensions": [float(dim) for dim in data[8:11]],
                        "Location": [float(loc) for loc in data[11:14]],
                        "Rotation": float(data[14]),
                    }
                    annotations.append(annotation)
        return annotations
    return None


def crop_image_around_point(image, x, y, offset):
    """Crop a square patch around a point, padding missing regions with zeros."""
    height, width = image.shape[:2]
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(width, x + offset)
    y2 = min(height, y + offset)

    cropped_height = 2 * offset
    cropped_width = 2 * offset
    if x1 >= x2 or y1 >= y2:
        return np.zeros((cropped_height, cropped_width, image.shape[2]), dtype=image.dtype)

    cropped_image = np.zeros((cropped_height, cropped_width, image.shape[2]), dtype=image.dtype)

    offset_x1 = max(0, offset - (x - x1))
    offset_y1 = max(0, offset - (y - y1))
    offset_x2 = min(2 * offset, offset_x1 + (x2 - x1))
    offset_y2 = min(2 * offset, offset_y1 + (y2 - y1))

    cropped_image[offset_y1:offset_y2, offset_x1:offset_x2] = image[y1:y2, x1:x2]
    return cropped_image


def project_3d_to_2d(points_3d, fx, fy, cx, cy):
    """Project 3D points in camera coordinates onto the image plane."""
    points_2d = []
    for point in points_3d:
        x_coord, y_coord, z_coord = point
        x = (fx * x_coord / z_coord) + cx
        y = (fy * y_coord / z_coord) + cy
        points_2d.append((x, y))
    return np.array(points_2d)


def filter_vlp16(points):
    """Filter invalid or backward-facing VLP-16 LiDAR points."""
    new_pcl = []
    cnt_zero = 0
    cnt_negative_x = 0
    for point in points:
        if point[3] == 0:
            cnt_zero += 1
            continue
        elif point[0] < 0:
            cnt_negative_x += 1
        else:
            new_pcl.append(point)
    return np.array(new_pcl)


def visualize_with_image_color(image, points, color):
    """Draw 2D points on an image with a fixed color."""
    for i in range(len(points)):
        cv2.circle(image, (int(points[i, 0]), int(points[i, 1])), 3, color=color, thickness=-1)
    return image


def visulize_vod(pcd_radar, image, radar_to_camera, intrinsic, color):
    """Project radar points onto the image and visualize them."""
    geo_radar = pcd_radar[:, 0:3]
    points_3d_radar_to_cam = trans_point_coor(geo_radar, radar_to_camera)
    fx, cx = intrinsic[0, 0], intrinsic[0, 2]
    fy, cy = intrinsic[1, 1], intrinsic[1, 2]
    points_2d_radar = project_3d_to_2d(points_3d_radar_to_cam, fx, fy, cx, cy)
    new_image = visualize_with_image_color(image, points_2d_radar, color)
    cv2.imshow("new", new_image)
    cv2.waitKey(0)


def get_vod_dir(frame):
    """Return commonly used VoD file paths for a given frame index."""
    cam_base_dir = os.path.join(VOD_RADAR_TRAIN_ROOT, "image_2")
    radar_base_dir = os.path.join(VOD_RADAR_TRAIN_ROOT, "velodyne")
    calib_base_dir = os.path.join(VOD_RADAR_TRAIN_ROOT, "calib")
    lidar_base_dir = os.path.join(VOD_LIDAR_TRAIN_ROOT, "velodyne")
    lidar_calib_base_dir = os.path.join(VOD_LIDAR_TRAIN_ROOT, "calib")

    frame_str = str(frame).zfill(5)
    radar_file = os.path.join(radar_base_dir, f"{frame_str}.bin")
    lidar_file = os.path.join(lidar_base_dir, f"{frame_str}.bin")
    cam_file = os.path.join(cam_base_dir, f"{frame_str}.jpg")
    calib_file = os.path.join(calib_base_dir, f"{frame_str}.txt")
    lidar_calib_file = os.path.join(lidar_calib_base_dir, f"{frame_str}.txt")
    txt_base_dir = VOD_LABEL_ROOT + os.sep

    return radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir


def crop_by_bbox(points, bbox_3d):
    """Crop points inside an axis-aligned 3D bounding box."""
    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3d
    x_indices = np.logical_and(points[:, 0] >= xmin, points[:, 0] <= xmax)
    y_indices = np.logical_and(points[:, 1] >= ymin, points[:, 1] <= ymax)
    z_indices = np.logical_and(points[:, 2] >= zmin, points[:, 2] <= zmax)
    indices_within_bbox = np.logical_and.reduce((x_indices, y_indices, z_indices))
    return points[indices_within_bbox]


def read_3dbbox(annotation):
    """Convert an annotation entry into an axis-aligned 3D bounding box."""
    x, y, z = annotation["Location"]
    length, width, height = annotation["Dimensions"]
    xmin, xmax = x - length / 2, x + length / 2
    ymin, ymax = y - width / 2, y + width / 2
    zmin, zmax = z - height / 2, z + height / 2
    bbox_3d = [xmin, xmax, ymin, ymax, zmin, zmax]
    loc = [x, y, z]
    return bbox_3d, loc


def read_pose(frame_number):
    """Read pose matrices for a frame from a VoD pose JSONL file."""
    file_path = VOD_POSE_TEMPLATE.format(frame_number)
    if not os.path.exists(file_path):
        return None, None, None

    odom_to_camera = None
    map_to_camera = None
    utm_to_camera = None

    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if "odomToCamera" in data:
                odom_to_camera = np.array(data["odomToCamera"]).reshape(4, 4)
            elif "mapToCamera" in data:
                map_to_camera = np.array(data["mapToCamera"]).reshape(4, 4)
            elif "UTMToCamera" in data:
                utm_to_camera = np.array(data["UTMToCamera"]).reshape(4, 4)

    return odom_to_camera, map_to_camera, utm_to_camera


def compute_transform(frame0, frame1, radar_to_camera0, radar_to_camera1):
    """Compute frame-to-frame transforms in the radar coordinate system."""
    odom_to_camera0, map_to_camera0, utm_to_camera0 = read_pose(frame0)
    odom_to_camera1, map_to_camera1, utm_to_camera1 = read_pose(frame1)

    if odom_to_camera0 is None or odom_to_camera1 is None:
        print("odom_to_camera not found.")
        return None, None, None

    radar_to_camera0_inv = np.linalg.inv(radar_to_camera0)

    odom_transform_camera = np.dot(np.linalg.inv(odom_to_camera0), odom_to_camera1)
    odom_transform_radar = np.dot(np.dot(radar_to_camera0_inv, odom_transform_camera), radar_to_camera1)

    map_transform_radar = None
    utm_transform_radar = None
    if (
        map_to_camera0 is not None
        and utm_to_camera0 is not None
        and map_to_camera1 is not None
        and utm_to_camera1 is not None
    ):
        map_transform_camera = np.dot(np.linalg.inv(map_to_camera0), map_to_camera1)
        utm_transform_camera = np.dot(np.linalg.inv(utm_to_camera0), utm_to_camera1)

        map_transform_radar = np.dot(np.dot(radar_to_camera0_inv, map_transform_camera), radar_to_camera1)
        utm_transform_radar = np.dot(np.dot(radar_to_camera0_inv, utm_transform_camera), radar_to_camera1)

    return odom_transform_radar, map_transform_radar, utm_transform_radar


def transform_bbox_to_radar(bbox, radar_to_camera):
    """Transform an axis-aligned camera-frame bounding box into the radar frame."""
    camera_to_radar = np.linalg.inv(radar_to_camera)

    corners = np.array([
        [bbox[0], bbox[2], bbox[4], 1],
        [bbox[0], bbox[2], bbox[5], 1],
        [bbox[0], bbox[3], bbox[4], 1],
        [bbox[0], bbox[3], bbox[5], 1],
        [bbox[1], bbox[2], bbox[4], 1],
        [bbox[1], bbox[2], bbox[5], 1],
        [bbox[1], bbox[3], bbox[4], 1],
        [bbox[1], bbox[3], bbox[5], 1],
    ])

    radar_corners = (np.dot(camera_to_radar, corners.T)).T
    x_coords = radar_corners[:, 0]
    y_coords = radar_corners[:, 1]
    z_coords = radar_corners[:, 2]

    radar_bbox = [
        x_coords.min(),
        x_coords.max(),
        y_coords.min(),
        y_coords.max(),
        z_coords.min(),
        z_coords.max(),
    ]
    return radar_bbox


def transform_coor_to_radar(location, radar_transform_matrix):
    """Transform a 3D point from camera coordinates to radar coordinates."""
    transform_inv = np.linalg.inv(radar_transform_matrix)
    location_homogeneous = np.array([location[0], location[1], location[2], 1])
    location_radar_homogeneous = np.dot(transform_inv, location_homogeneous)
    return location_radar_homogeneous[:3]


def filter_points_in_bbox(point_cloud, bbox_location, bbox_dimensions, bbox_yaw):
    """Filter points inside a rotated 3D bounding box around the z-axis."""
    bbox_center = np.array(bbox_location)
    bbox_height, bbox_length, bbox_width = bbox_dimensions
    yaw = bbox_yaw

    rotation_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])

    points_xyz = point_cloud[:, :3]
    translated_points = points_xyz - bbox_center
    local_points = np.dot(translated_points, rotation_z.T)

    mask = (
        (local_points[:, 0] >= -bbox_length / 2)
        & (local_points[:, 0] <= bbox_length / 2)
        & (local_points[:, 1] >= -bbox_width / 2)
        & (local_points[:, 1] <= bbox_width / 2)
        & (local_points[:, 2] >= -bbox_height / 2)
        & (local_points[:, 2] <= bbox_height / 2)
    )

    return point_cloud[mask], mask


def cartesian_to_spherical(vector):
    """Convert a Cartesian vector to magnitude, pitch, and yaw (in degrees)."""
    vx, vy, vz = vector
    magnitude = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    pitch = np.arctan2(vz, np.sqrt(vx ** 2 + vy ** 2)) * 180 / np.pi
    yaw = np.arctan2(vy, vx) * 180 / np.pi
    return magnitude, pitch, yaw


# ============================================================
# Utilities for the Astyx dataset
# ============================================================
def get_asytx_dir(frame):
    """Return commonly used Astyx file paths for a given frame index."""
    cam_base_dir = os.path.join(ASTYX_ROOT, "camera_front")
    radar_base_dir = os.path.join(ASTYX_ROOT, "radar_6455")
    calib_base_dir = os.path.join(ASTYX_ROOT, "calibration")
    object_base_dir = os.path.join(ASTYX_ROOT, "groundtruth_obj3d")
    lidar_base_dir = os.path.join(ASTYX_ROOT, "lidar_vlp16")

    frame_str = str(frame).zfill(6)
    radar_file = os.path.join(radar_base_dir, f"{frame_str}.txt")
    lidar_file = os.path.join(lidar_base_dir, f"{frame_str}.txt")
    cam_file = os.path.join(cam_base_dir, f"{frame_str}.jpg")
    calib_file = os.path.join(calib_base_dir, f"{frame_str}.json")
    object_file = os.path.join(object_base_dir, f"{frame_str}.json")
    return radar_file, lidar_file, cam_file, calib_file, object_file


def get_transform_matrix(calib_file, sensor_uid_A, sensor_uid_B):
    """Compute the transform matrix from sensor A to sensor B using Astyx calibration."""
    with open(calib_file, "r") as file:
        calibration_data = json.load(file)

    transform_a_to_ref = None
    transform_b_to_ref = None

    for sensor in calibration_data["sensors"]:
        if sensor["sensor_uid"] == sensor_uid_A:
            transform_a_to_ref = np.array(sensor["calib_data"]["T_to_ref_COS"])
        elif sensor["sensor_uid"] == sensor_uid_B:
            transform_b_to_ref = np.array(sensor["calib_data"]["T_to_ref_COS"])

    if transform_a_to_ref is None or transform_b_to_ref is None:
        return None

    transform_a_to_b = np.linalg.inv(transform_b_to_ref) @ transform_a_to_ref
    return transform_a_to_b


def read_txt_data(file_path):
    """Read Astyx radar or LiDAR text data and remove NaN rows."""
    txt_data = np.genfromtxt(file_path, delimiter=" ", skip_header=1)
    txt_data = txt_data[~np.isnan(txt_data).any(axis=1)]
    return txt_data


if __name__ == "__main__":
    print(rpy_to_rotation_matrix([-0.3, 0.5, 0.1]))
