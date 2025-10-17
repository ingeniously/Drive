import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.color_map import get_colormap
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

# ---------------- CONFIG ----------------
VERSION = 'v1.0-trainval'
DATAROOT = '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Drive/LLaMA-Factory/data/nuscenes'
OUTDIR = '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Drive/LLaMA-Factory/data/rendered_boxes'
os.makedirs(OUTDIR, exist_ok=True)

CAM_CHANNELS = ['CAM_FRONT']

# Toggles
DRAW_LABELS = False          # Set True if you later want text again
RENDER_LANES = True          # Enable lane overlay
LANE_DIVIDER_COLOR = (0, 0, 255)     # Red in BGR for lane dividers
ROAD_DIVIDER_COLOR = (0, 0, 255)     # Red in BGR for road dividers  
ROAD_BORDER_COLOR = (0, 0, 255)      # Red in BGR for road borders
LANE_THICKNESS = 3
LANE_RADIUS = 80.0           # Increased radius to catch more lanes
MIN_LANE_POINTS = 2          # skip degenerate lanes
# ----------------------------------------

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
colormap = get_colormap()

# Cache NuScenesMap objects by location name
_map_cache = {}

def get_nusc_map(location: str):
    """Get or create cached NuScenesMap for a location."""
    if location not in _map_cache:
        try:
            _map_cache[location] = NuScenesMap(dataroot=DATAROOT, map_name=location)
        except Exception as e:
            print(f"Could not load map for {location}: {e}")
            return None
    return _map_cache[location]

def project_points(points_3d, K):
    """
    points_3d: (3, N) camera frame
    K: (3,3) intrinsic
    Returns: (N,2) pixel coords, depths (N,)
    """
    pts = K @ points_3d
    zs = pts[2, :]
    # Avoid division by zero
    zs[zs == 0] = 1e-6
    xs = pts[0, :] / zs
    ys = pts[1, :] / zs
    return np.stack([xs, ys], axis=1), zs

def draw_box(img, corners_3d, K, color=(0,255,0), thickness=5):
    """Draw a 3D bounding box on the image."""
    pts_2d, depth = project_points(corners_3d, K)
    if (depth <= 0).all():
        return img
    pts_2d = pts_2d.astype(int)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        cv2.line(img, tuple(pts_2d[i]), tuple(pts_2d[j]), color, thickness, cv2.LINE_AA)
    return img

def category_to_color(category_name):
    """Get color for a category from the colormap."""
    return tuple(int(c) for c in colormap.get(category_name, (0,255,0)))

def world_to_ego(points_xyz, ego_pose_rec):
    """
    Transform points from world frame to ego vehicle frame.
    points_xyz: (N,3) world coords
    ego_pose_rec: record with translation, rotation (quat)
    Returns: (N,3) points in ego frame
    """
    ego_rot = Quaternion(ego_pose_rec['rotation'])
    ego_trans = np.array(ego_pose_rec['translation'])
    # World -> ego vehicle
    points_ego = (ego_rot.inverse.rotation_matrix @ (points_xyz - ego_trans).T).T
    return points_ego

def ego_to_camera(points_ego, calib_sensor_rec):
    """
    Transform points from ego vehicle frame to camera frame.
    points_ego: (N,3) ego coords
    calib_sensor_rec: calibrated sensor with translation, rotation (quat)
    Returns: (3, N) points in camera frame
    """
    sens_rot = Quaternion(calib_sensor_rec['rotation'])
    sens_trans = np.array(calib_sensor_rec['translation'])
    # Ego -> sensor
    pts_cam = (sens_rot.inverse.rotation_matrix @ (points_ego - sens_trans).T)
    return pts_cam  # (3,N)

def get_lane_lines(nusc_map, ego_x, ego_y, radius):
    """
    Get all lane dividers, road dividers, and road segment boundaries near ego position.
    Returns list of tuples: (line_coords, layer_type) where line_coords is (N, 2) array.
    """
    lines = []
    
    # LINE-BASED layers (dividers)
    line_layers = ['lane_divider', 'road_divider']
    
    # POLYGON-BASED layers (road segments for borders)
    polygon_layers = ['road_segment']
    
    try:
        # Get nearby line records
        line_records = nusc_map.get_records_in_radius(ego_x, ego_y, radius, line_layers)
        
        # Get nearby polygon records
        polygon_records = nusc_map.get_records_in_radius(ego_x, ego_y, radius, polygon_layers)
    except Exception as e:
        print(f"Error getting records in radius: {e}")
        return lines
    
    # Process line-based layers (lane_divider, road_divider)
    for layer_name in line_layers:
        if layer_name not in line_records:
            continue
        
        for token in line_records[layer_name]:
            try:
                record = nusc_map.get(layer_name, token)
                
                if 'node_tokens' in record:
                    node_tokens = record['node_tokens']
                    coords = []
                    for node_token in node_tokens:
                        node = nusc_map.get('node', node_token)
                        coords.append([node['x'], node['y']])
                    
                    if len(coords) >= MIN_LANE_POINTS:
                        lines.append((np.array(coords), layer_name))
                        
            except Exception as e:
                continue
    
    # Process road segments to extract borders
    # The key is that road_segment records have 'polygon_token', not 'polygon'
    if 'road_segment' in polygon_records:
        for token in polygon_records['road_segment']:
            try:
                record = nusc_map.get('road_segment', token)
                
                # Check if 'polygon_token' exists (not 'polygon')
                if 'polygon_token' in record:
                    polygon_token = record['polygon_token']
                    
                    # Get the polygon record
                    polygon_record = nusc_map.get('polygon', polygon_token)
                    
                    # Get exterior nodes using 'exterior_node_tokens' or 'node_tokens'
                    node_tokens = None
                    if 'exterior_node_tokens' in polygon_record:
                        node_tokens = polygon_record['exterior_node_tokens']
                    elif 'node_tokens' in polygon_record:
                        node_tokens = polygon_record['node_tokens']
                    
                    if node_tokens:
                        coords = []
                        for node_token in node_tokens:
                            node = nusc_map.get('node', node_token)
                            coords.append([node['x'], node['y']])
                        
                        # Close the polygon by adding first point at the end
                        if len(coords) >= MIN_LANE_POINTS:
                            coords.append(coords[0])
                            lines.append((np.array(coords), 'road_border'))
                            
            except Exception as e:
                continue
    
    lane_count = sum(1 for _, t in lines if t == 'lane_divider')
    road_count = sum(1 for _, t in lines if t == 'road_divider')
    border_count = sum(1 for _, t in lines if t == 'road_border')
    
    return lines

def draw_lanes_on_image(img, ego_pose_rec, calib_sensor_rec, cam_intrinsic, nusc_map,
                        lane_radius=80.0, lane_thickness=3):
    """
    Projects lane dividers, road dividers, and road borders into the camera image.
    """
    if nusc_map is None:
        return img

    ego_x, ego_y, ego_z = ego_pose_rec['translation']
    
    # Get all lane lines near ego
    lane_data = get_lane_lines(nusc_map, ego_x, ego_y, lane_radius)
    
    if not lane_data:
        return img
    
    cam_K = np.array(cam_intrinsic)
    h, w = img.shape[:2]
    
    total_lines_drawn = 0
    
    # Process each lane line/border
    for lane_xy, layer_type in lane_data:
        try:
            # Choose color based on layer type
            if layer_type == 'lane_divider':
                color = LANE_DIVIDER_COLOR
            elif layer_type == 'road_divider':
                color = ROAD_DIVIDER_COLOR
            elif layer_type == 'road_border':
                color = ROAD_BORDER_COLOR
            else:
                color = (0, 0, 255)  # Default red
            
            # lane_xy is (N, 2) in world frame (x, y)
            # Add z=0 (ground level)
            num_points = len(lane_xy)
            pts_world = np.column_stack([lane_xy, np.zeros(num_points)])
            
            # Transform: world -> ego -> camera
            pts_ego = world_to_ego(pts_world, ego_pose_rec)
            pts_cam = ego_to_camera(pts_ego, calib_sensor_rec)  # (3, N)
            
            # Filter points that are behind the camera
            depths = pts_cam[2, :]
            valid_mask = depths > 0.5
            
            if np.sum(valid_mask) < MIN_LANE_POINTS:
                continue
            
            pts_cam_valid = pts_cam[:, valid_mask]
            
            # Project to 2D image plane
            pix, _ = project_points(pts_cam_valid, cam_K)
            pix_int = pix.astype(int)
            
            # Draw the polyline
            for i in range(len(pix_int) - 1):
                p1 = tuple(pix_int[i])
                p2 = tuple(pix_int[i+1])
                
                # Draw even if slightly outside to show continuity
                if (-w <= p1[0] <= 2*w and -h <= p1[1] <= 2*h and
                    -w <= p2[0] <= 2*w and -h <= p2[1] <= 2*h):
                    cv2.line(img, p1, p2, color, lane_thickness, cv2.LINE_AA)
                    total_lines_drawn += 1
        except Exception as e:
            continue
    
    return img

def process_all():
    """Process all camera images and render boxes and lanes."""
    camera_sd_tokens = [
        sd['token'] for sd in nusc.sample_data
        if sd['is_key_frame'] and sd['channel'] in CAM_CHANNELS
    ]
    
    print(f"Processing {len(camera_sd_tokens)} images...")
    
    # Process all images
    for idx, token in enumerate(tqdm(camera_sd_tokens, desc='Rendering')):
        sd_rec = nusc.get('sample_data', token)
        sample_rec = nusc.get('sample', sd_rec['sample_token'])
        scene_rec = nusc.get('scene', sample_rec['scene_token'])
        log_rec = nusc.get('log', scene_rec['log_token'])
        location = log_rec['location']
        
        nusc_map = get_nusc_map(location)

        data_path, boxes, cam_intrinsic = nusc.get_sample_data(
            token,
            use_flat_vehicle_coordinates=False
        )
        img = cv2.imread(data_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        ego_pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        calib_sensor_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        cam_K = np.array(cam_intrinsic)

        # Draw lanes first as background
        if RENDER_LANES and nusc_map is not None:
            img = draw_lanes_on_image(img, ego_pose_rec, calib_sensor_rec, cam_K, nusc_map,
                           lane_radius=LANE_RADIUS, lane_thickness=LANE_THICKNESS)

        # Draw 3D boxes on top
        for box in boxes:
            corners_in_cam = box.corners()
            color = category_to_color(box.name)
            draw_box(img, corners_in_cam, cam_K, color=color, thickness=5)

            if DRAW_LABELS:
                center_in_cam = box.center.reshape(3,1)
                center_2d, _ = project_points(center_in_cam, cam_K)
                cpt = tuple(center_2d[0].astype(int))
                if 0 <= cpt[0] < w and 0 <= cpt[1] < h:
                    cv2.putText(img, box.name.split('.')[-1], cpt,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Create structured output directory
        dir_type = 'samples' if sd_rec['is_key_frame'] else 'sweeps'
        channel_name = sd_rec['channel']
        
        output_subdir = os.path.join(OUTDIR, dir_type, channel_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        output_path = os.path.join(output_subdir, os.path.basename(sd_rec['filename']))
        cv2.imwrite(output_path, img)

process_all()
print("\nDone.")