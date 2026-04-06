import os
import glob
import math
import json
import numpy as np
import pandas as pd
import open3d as o3d
import utils

# =========================================================
# User settings
# =========================================================
FRAME_DT = 0.1
TOTAL_FRAMES = 9931

OUTPUT_DYNAMIC_CSV = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/dynamic_objects_total.csv'
OUTPUT_EGO_CSV = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/ego_velocity_lidar_icp.csv'
OUTPUT_EGO_NPY = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/radar_velo_lidar_icp.npy'

DEBUG_DIR = '/workspace/data/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/debug_local_icp'
os.makedirs(DEBUG_DIR, exist_ok=True)

# -------------------------
# thresholds
# -------------------------
INVALID_CLASSES = {
    "DontCare",
    "bicycle_rack",
    "human_depiction",
}

# odom suspicious pair thresholds
MAX_ODOM_TRANS_NORM = 8.0        # meters
MAX_ODOM_YAW_DEG = 20.0          # deg
MAX_ODOM_PITCH_ROLL_DEG = 10.0   # deg

# ego thresholds
MAX_EGO_SPEED = 20.0             # m/s, beyond this set to 0

# object thresholds
MAX_OBJECT_SPEED = 25.0          # m/s
MAX_RADIAL_SPEED = 20.0          # m/s
MAX_ALIGNMENT_RESID = 3.5        # meters
DIRECT_FALLBACK_MARGIN = 0.8     # if direct is much better than repo, use it

# ICP thresholds
USE_LOCAL_ICP = True
ICP_MIN_POINTS = 500
ICP_VOXEL_SIZE = 0.4
ICP_COARSE_DIST = 2.0
ICP_FINE_DIST = 0.8
ICP_MIN_FITNESS = 0.35
ICP_MAX_RMSE = 1.2
ICP_MAX_DELTA_TRANS = 6.0        # max correction w.r.t odom init
ICP_MAX_DELTA_ROT_DEG = 20.0

# lidar spatial filter
LIDAR_X_MIN = -5.0
LIDAR_X_MAX = 80.0
LIDAR_Y_ABS = 40.0
LIDAR_Z_MIN = -3.5
LIDAR_Z_MAX = 5.0

# optional hard skip pairs from your diagnostics
HARD_SKIP_PAIRS = {
    (1311, 1312),
    (1312, 1313),
}

# cache
_frame_cache = {}
_pair_cache = {}

# =========================================================
# basic helpers
# =========================================================
def trans_point(T, p):
    p4 = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T @ p4
    return out[:3]

def rotation_metrics_from_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]

    trans_norm = float(np.linalg.norm(t))
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord='fro'))
    det = float(np.linalg.det(R))

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return {
        "trans_norm": trans_norm,
        "orth_err": orth_err,
        "det": det,
        "yaw_deg": float(np.degrees(yaw)),
        "pitch_deg": float(np.degrees(pitch)),
        "roll_deg": float(np.degrees(roll)),
    }

def relative_rotation_angle_deg(T_a, T_b):
    R = T_a[:3, :3] @ T_b[:3, :3].T
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))

def get_available_frames():
    _, _, _, _, _, txt_base_dir = utils.get_vod_dir(0)
    label_files = sorted(glob.glob(os.path.join(txt_base_dir, "*.txt")))
    return [int(os.path.splitext(os.path.basename(f))[0]) for f in label_files]

def load_frame_info(frame):
    if frame in _frame_cache:
        return _frame_cache[frame]

    radar_file, lidar_file, cam_file, calib_file, lidar_calib_file, txt_base_dir = utils.get_vod_dir(frame)
    label_file = os.path.join(txt_base_dir, f"{frame:05d}.txt")

    if not os.path.exists(calib_file) or not os.path.exists(label_file):
        _frame_cache[frame] = None
        return None

    annotations = utils.read_annotation(label_file)
    if annotations is None:
        _frame_cache[frame] = None
        return None

    T_radar = utils.get_radar2cam(calib_file)
    T_lidar = utils.get_lidar2cam(lidar_calib_file) if os.path.exists(lidar_calib_file) else None

    if T_radar is None or T_lidar is None:
        _frame_cache[frame] = None
        return None

    track_to_anno = {anno["Track_ID"]: anno for anno in annotations}

    data = {
        "frame": frame,
        "radar_file": radar_file,
        "lidar_file": lidar_file,
        "calib_file": calib_file,
        "lidar_calib_file": lidar_calib_file,
        "label_file": label_file,
        "T_radar": T_radar,
        "T_lidar": T_lidar,
        "annotations": annotations,
        "track_to_anno": track_to_anno,
    }
    _frame_cache[frame] = data
    return data

def anno_to_radar_state(anno, T_radar):
    bbox_cam, loc_cam = utils.read_3dbbox(anno)
    bbox_radar = utils.transform_bbox_to_radar(bbox_cam, T_radar)
    loc_radar = np.array(utils.transform_coor_to_radar(loc_cam, T_radar), dtype=np.float64)

    xmin, xmax, ymin, ymax, zmin, zmax = bbox_radar
    dim_x = xmax - xmin
    dim_y = ymax - ymin
    dim_z = zmax - zmin

    return {
        "loc_radar": loc_radar,
        "bbox_radar": np.array(bbox_radar, dtype=np.float64),
        "dim_xyz": np.array([dim_x, dim_y, dim_z], dtype=np.float64),
    }

# =========================================================
# lidar loading / preprocessing
# =========================================================
def lidar_to_radar_transform(T_radar, T_lidar):
    # point_cam = T_lidar @ point_lidar
    # point_radar = inv(T_radar) @ point_cam
    return np.linalg.inv(T_radar) @ T_lidar

def load_lidar_points_in_radar(frame, remove_boxes=True, box_expand=0.5):
    info = load_frame_info(frame)
    if info is None:
        return None

    lidar_file = info["lidar_file"]
    if not os.path.exists(lidar_file):
        return None

    raw = np.fromfile(lidar_file, dtype=np.float32)
    if raw.size % 4 != 0:
        return None
    raw = raw.reshape(-1, 4)
    xyz_lidar = raw[:, :3].astype(np.float64)

    T_lidar_to_radar = lidar_to_radar_transform(info["T_radar"], info["T_lidar"])
    xyz_radar = utils.trans_point_coor(xyz_lidar, T_lidar_to_radar)

    # spatial filter
    mask = (
        (xyz_radar[:, 0] >= LIDAR_X_MIN) &
        (xyz_radar[:, 0] <= LIDAR_X_MAX) &
        (np.abs(xyz_radar[:, 1]) <= LIDAR_Y_ABS) &
        (xyz_radar[:, 2] >= LIDAR_Z_MIN) &
        (xyz_radar[:, 2] <= LIDAR_Z_MAX)
    )
    xyz_radar = xyz_radar[mask]

    if remove_boxes and len(info["annotations"]) > 0:
        keep_mask = np.ones(len(xyz_radar), dtype=bool)
        for anno in info["annotations"]:
            bbox_radar = utils.transform_bbox_to_radar(utils.read_3dbbox(anno)[0], info["T_radar"])
            xmin, xmax, ymin, ymax, zmin, zmax = bbox_radar
            xmin -= box_expand
            xmax += box_expand
            ymin -= box_expand
            ymax += box_expand
            zmin -= box_expand
            zmax += box_expand

            inside = (
                (xyz_radar[:, 0] >= xmin) & (xyz_radar[:, 0] <= xmax) &
                (xyz_radar[:, 1] >= ymin) & (xyz_radar[:, 1] <= ymax) &
                (xyz_radar[:, 2] >= zmin) & (xyz_radar[:, 2] <= zmax)
            )
            keep_mask &= (~inside)
        xyz_radar = xyz_radar[keep_mask]

    return xyz_radar

def make_pcd(points_xyz, voxel_size):
    if points_xyz is None or len(points_xyz) < ICP_MIN_POINTS:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) < ICP_MIN_POINTS:
        return None
    radius = voxel_size * 2.5
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return pcd

# =========================================================
# transform selection
# =========================================================
def compute_repo_odom_transform(prev_frame, curr_frame, T_radar_prev, T_radar_curr):
    # repo convention: returned T is "curr -> prev"
    odom_T_repo, map_T_repo, utm_T_repo = utils.compute_transform(
        prev_frame, curr_frame, T_radar_prev, T_radar_curr
    )
    return odom_T_repo

def odom_is_suspicious(T_repo):
    if T_repo is None:
        return True, "missing_odom"

    T_prev_to_curr = np.linalg.inv(T_repo)
    m = rotation_metrics_from_transform(T_prev_to_curr)

    if m["trans_norm"] > MAX_ODOM_TRANS_NORM:
        return True, "odom_trans_too_large"
    if abs(m["yaw_deg"]) > MAX_ODOM_YAW_DEG:
        return True, "odom_yaw_too_large"
    if max(abs(m["pitch_deg"]), abs(m["roll_deg"])) > MAX_ODOM_PITCH_ROLL_DEG:
        return True, "odom_pitch_roll_too_large"
    return False, "ok"

def run_local_icp(prev_frame, curr_frame, T_prev_to_curr_init):
    prev_pts = load_lidar_points_in_radar(prev_frame, remove_boxes=True)
    curr_pts = load_lidar_points_in_radar(curr_frame, remove_boxes=True)

    if prev_pts is None or curr_pts is None:
        return None, {"icp_status": "missing_lidar"}

    src = make_pcd(prev_pts, ICP_VOXEL_SIZE)
    tgt = make_pcd(curr_pts, ICP_VOXEL_SIZE)

    if src is None or tgt is None:
        return None, {"icp_status": "not_enough_points"}

    try:
        reg_coarse = o3d.pipelines.registration.registration_icp(
            src, tgt, ICP_COARSE_DIST, T_prev_to_curr_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        reg_fine = o3d.pipelines.registration.registration_icp(
            src, tgt, ICP_FINE_DIST, reg_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    except Exception as e:
        return None, {"icp_status": f"icp_exception:{str(e)}"}

    T_icp = reg_fine.transformation
    fitness = float(reg_fine.fitness)
    rmse = float(reg_fine.inlier_rmse)

    delta_trans = float(np.linalg.norm(
        T_icp[:3, 3] - T_prev_to_curr_init[:3, 3]
    ))
    delta_rot_deg = relative_rotation_angle_deg(T_icp, T_prev_to_curr_init)

    accepted = (
        fitness >= ICP_MIN_FITNESS and
        rmse <= ICP_MAX_RMSE and
        delta_trans <= ICP_MAX_DELTA_TRANS and
        delta_rot_deg <= ICP_MAX_DELTA_ROT_DEG
    )

    debug = {
        "icp_status": "accepted" if accepted else "rejected",
        "icp_fitness": fitness,
        "icp_rmse": rmse,
        "icp_delta_trans": delta_trans,
        "icp_delta_rot_deg": delta_rot_deg,
        "src_points": len(src.points),
        "tgt_points": len(tgt.points),
    }
    return (T_icp if accepted else None), debug

def get_pair_transform(prev_frame, curr_frame):
    """
    Return:
        {
            'ok': bool,
            'T_repo': current->previous   (repo convention)
            'T_prev_to_curr': previous->current
            'pair_mode': 'odom' or 'icp' or 'zero'
            'pair_status': ...
            ...
        }
    """
    key = (prev_frame, curr_frame)
    if key in _pair_cache:
        return _pair_cache[key]

    result = {
        "ok": False,
        "T_repo": None,
        "T_prev_to_curr": None,
        "pair_mode": "zero",
        "pair_status": "unknown",
        "icp_status": "not_run",
        "icp_fitness": np.nan,
        "icp_rmse": np.nan,
        "icp_delta_trans": np.nan,
        "icp_delta_rot_deg": np.nan,
    }

    if key in HARD_SKIP_PAIRS:
        result["pair_status"] = "hard_skip_pair"
        _pair_cache[key] = result
        return result

    prev_info = load_frame_info(prev_frame)
    curr_info = load_frame_info(curr_frame)
    if prev_info is None or curr_info is None:
        result["pair_status"] = "missing_frame_info"
        _pair_cache[key] = result
        return result

    T_repo_odom = compute_repo_odom_transform(
        prev_frame, curr_frame, prev_info["T_radar"], curr_info["T_radar"]
    )
    suspicious, reason = odom_is_suspicious(T_repo_odom)

    # case 1: odom good -> use odom directly
    if T_repo_odom is not None and not suspicious:
        result["ok"] = True
        result["T_repo"] = T_repo_odom
        result["T_prev_to_curr"] = np.linalg.inv(T_repo_odom)
        result["pair_mode"] = "odom"
        result["pair_status"] = "odom_ok"
        _pair_cache[key] = result
        return result

    # case 2: odom suspicious, try local ICP refinement
    if USE_LOCAL_ICP and T_repo_odom is not None:
        T_prev_to_curr_init = np.linalg.inv(T_repo_odom)
        T_prev_to_curr_icp, debug = run_local_icp(prev_frame, curr_frame, T_prev_to_curr_init)

        result.update(debug)

        if T_prev_to_curr_icp is not None:
            result["ok"] = True
            result["T_prev_to_curr"] = T_prev_to_curr_icp
            result["T_repo"] = np.linalg.inv(T_prev_to_curr_icp)
            result["pair_mode"] = "icp"
            result["pair_status"] = f"odom_suspicious_but_icp_fixed:{reason}"
            _pair_cache[key] = result
            return result

    # case 3: odom missing or suspicious and icp failed -> zero
    if T_repo_odom is None:
        result["pair_status"] = "missing_odom_zero"
    else:
        result["pair_status"] = f"suspicious_odom_zero:{reason}"

    _pair_cache[key] = result
    return result

# =========================================================
# ego velocity
# =========================================================
def compute_ego_velocity_for_current_frame(curr_frame):
    """
    Ego velocity in CURRENT radar frame.
    Uses pair (curr-1 -> curr), if missing returns zero.
    """
    if curr_frame <= 0:
        return np.zeros(3, dtype=np.float64), "frame0_zero", None

    pair = get_pair_transform(curr_frame - 1, curr_frame)
    if not pair["ok"]:
        return np.zeros(3, dtype=np.float64), pair["pair_status"], pair

    # T_prev_to_curr maps previous radar coordinates into current radar coordinates
    # translation is previous-origin in current coordinates
    t_prev_origin_in_curr = pair["T_prev_to_curr"][:3, 3]
    ego_v = -t_prev_origin_in_curr / FRAME_DT

    if np.linalg.norm(ego_v) > MAX_EGO_SPEED:
        return np.zeros(3, dtype=np.float64), "ego_speed_too_large_zero", pair

    return ego_v.astype(np.float64), pair["pair_mode"], pair

def build_ego_velocity_csv():
    rows = []
    ego_array = np.zeros((TOTAL_FRAMES, 3), dtype=np.float32)

    for frame in range(TOTAL_FRAMES):
        v, status, pair = compute_ego_velocity_for_current_frame(frame)
        ego_array[frame] = v.astype(np.float32)

        row = {
            "Frame": frame,
            "vx": float(v[0]),
            "vy": float(v[1]),
            "vz": float(v[2]),
            "status": status,
        }

        if pair is not None:
            row.update({
                "pair_mode": pair["pair_mode"],
                "pair_status": pair["pair_status"],
                "icp_status": pair["icp_status"],
                "icp_fitness": pair["icp_fitness"],
                "icp_rmse": pair["icp_rmse"],
            })
        else:
            row.update({
                "pair_mode": "zero",
                "pair_status": status,
                "icp_status": "not_run",
                "icp_fitness": np.nan,
                "icp_rmse": np.nan,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_EGO_CSV, index=False)
    np.save(OUTPUT_EGO_NPY, ego_array)

    print(f"Saved ego CSV: {OUTPUT_EGO_CSV}")
    print(f"Saved ego NPY: {OUTPUT_EGO_NPY}")
    print(df["status"].value_counts(dropna=False))
    return df, ego_array

# =========================================================
# object radial velocity
# =========================================================
def choose_prev_center_in_curr(loc_prev, loc_curr, T_repo, allow_direct_fallback=True):
    """
    repo_style:
        pred = transform_coor_to_radar(loc_prev, T_repo)
    direct_T:
        pred = T_repo @ loc_prev
    """
    pred_repo = np.array(utils.transform_coor_to_radar(loc_prev, T_repo), dtype=np.float64)
    resid_repo = float(np.linalg.norm(loc_curr - pred_repo))

    if not allow_direct_fallback:
        return pred_repo, "repo_style", resid_repo

    pred_direct = trans_point(T_repo, loc_prev)
    resid_direct = float(np.linalg.norm(loc_curr - pred_direct))

    if resid_direct + DIRECT_FALLBACK_MARGIN < resid_repo:
        return pred_direct, "direct_T_fallback", resid_direct

    return pred_repo, "repo_style", resid_repo

def velocity_vector_to_radial(v_xyz, loc_radar):
    norm = np.linalg.norm(loc_radar)
    if norm < 1e-8:
        return 0.0
    ray_unit = loc_radar / norm
    return float(np.dot(v_xyz, ray_unit))

def estimate_object_row(curr_frame, anno_curr):
    curr_info = load_frame_info(curr_frame)
    if curr_info is None:
        return None

    track_id = anno_curr["Track_ID"]
    cls_name = anno_curr["Class"]
    rotation = anno_curr["Rotation"]

    curr_state = anno_to_radar_state(anno_curr, curr_info["T_radar"])
    loc_curr = curr_state["loc_radar"]
    dim_xyz = curr_state["dim_xyz"]

    base_row = {
        "Frame": curr_frame,
        "Track_ID": track_id,
        "Class": cls_name,
        "Rotation": rotation,
        "Location_x": float(loc_curr[0]),
        "Location_y": float(loc_curr[1]),
        "Location_z": float(loc_curr[2]),
        "PointNum": 0,
        "Dimension_x": float(dim_xyz[0]),
        "Dimension_y": float(dim_xyz[1]),
        "Dimension_z": float(dim_xyz[2]),
        "v_r_real": 0.0,
        "status": "zero_default",
        "pair_mode": "zero",
        "align_mode": "none",
        "speed_3d": 0.0,
        "align_resid": np.nan,
    }

    if cls_name in INVALID_CLASSES:
        base_row["status"] = "invalid_class_zero"
        return base_row

    if curr_frame <= 0:
        base_row["status"] = "frame0_zero"
        return base_row

    prev_frame = curr_frame - 1
    prev_info = load_frame_info(prev_frame)
    if prev_info is None:
        base_row["status"] = "missing_prev_frame_zero"
        return base_row

    if track_id not in prev_info["track_to_anno"]:
        base_row["status"] = "missing_prev_track_zero"
        return base_row

    pair = get_pair_transform(prev_frame, curr_frame)
    base_row["pair_mode"] = pair["pair_mode"]

    if not pair["ok"]:
        base_row["status"] = f"bad_pair_zero:{pair['pair_status']}"
        return base_row

    anno_prev = prev_info["track_to_anno"][track_id]
    prev_state = anno_to_radar_state(anno_prev, prev_info["T_radar"])
    loc_prev = prev_state["loc_radar"]

    # allow direct fallback only when using odom; if icp already fixed the pair, keep repo-style
    allow_direct = (pair["pair_mode"] == "odom")

    loc_prev_in_curr, align_mode, resid = choose_prev_center_in_curr(
        loc_prev, loc_curr, pair["T_repo"], allow_direct_fallback=allow_direct
    )

    delta = loc_curr - loc_prev_in_curr
    v_xyz = delta / FRAME_DT
    speed_3d = float(np.linalg.norm(v_xyz))
    v_r_real = velocity_vector_to_radial(v_xyz, loc_curr)

    base_row["align_mode"] = align_mode
    base_row["align_resid"] = resid
    base_row["speed_3d"] = speed_3d

    if resid > MAX_ALIGNMENT_RESID:
        base_row["status"] = "alignment_resid_too_large_zero"
        return base_row

    if speed_3d > MAX_OBJECT_SPEED:
        base_row["status"] = "speed_3d_too_large_zero"
        return base_row

    if abs(v_r_real) > MAX_RADIAL_SPEED:
        base_row["status"] = "radial_speed_too_large_zero"
        return base_row

    base_row["v_r_real"] = float(v_r_real)
    base_row["status"] = "ok"
    return base_row

def build_dynamic_objects_total_csv():
    frames = get_available_frames()
    rows = []

    for frame in frames:
        curr_info = load_frame_info(frame)
        if curr_info is None:
            continue

        for anno in curr_info["annotations"]:
            row = estimate_object_row(frame, anno)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "Frame", "Track_ID", "Class", "Rotation",
        "Location_x", "Location_y", "Location_z",
        "PointNum",
        "Dimension_x", "Dimension_y", "Dimension_z",
        "v_r_real",
        "status", "pair_mode", "align_mode",
        "speed_3d", "align_resid",
    ])

    df.to_csv(OUTPUT_DYNAMIC_CSV, index=False)
    print(f"Saved dynamic CSV: {OUTPUT_DYNAMIC_CSV}")
    print(df["status"].value_counts(dropna=False))

    # save pair debug
    pair_rows = []
    for (prev_frame, curr_frame), pair in sorted(_pair_cache.items()):
        row = {
            "prev_frame": prev_frame,
            "curr_frame": curr_frame,
            "ok": pair["ok"],
            "pair_mode": pair["pair_mode"],
            "pair_status": pair["pair_status"],
            "icp_status": pair["icp_status"],
            "icp_fitness": pair["icp_fitness"],
            "icp_rmse": pair["icp_rmse"],
            "icp_delta_trans": pair["icp_delta_trans"],
            "icp_delta_rot_deg": pair["icp_delta_rot_deg"],
        }
        if pair["T_prev_to_curr"] is not None:
            m = rotation_metrics_from_transform(pair["T_prev_to_curr"])
            row.update(m)
        pair_rows.append(row)

    pd.DataFrame(pair_rows).to_csv(
        os.path.join(DEBUG_DIR, "pair_transform_debug.csv"), index=False
    )

    return df

# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    ego_df, ego_arr = build_ego_velocity_csv()
    dynamic_df = build_dynamic_objects_total_csv()
