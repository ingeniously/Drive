import numpy as np

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def generate_three_of_thoughts(data_dict, lateral_offset=0.5, lane_half_width=3.5):
    """
    Generate a tree-of-thought reasoning for trajectory planning with 3 options per waypoint.
    Each option is evaluated on safety, smoothness, and efficiency with a weighted score.
    Returns structured output and final chosen trajectory.
    """
    fut = np.array(data_dict['gt_ego_fut_trajs'], dtype=float)  # [>=7, 2], we use idx 1..6
    his = np.array(data_dict['gt_ego_his_trajs'], dtype=float)  # [5, 2]
    prev = his[-1]  # previous point before the first future step
    
    # Get mission intent for scoring alignment
    cmd_vec = data_dict['gt_ego_fut_cmd']
    intent_direction = 0  # 0=forward, -1=left, 1=right
    if cmd_vec[0] > 0:  # right
        intent_direction = 1
    elif cmd_vec[1] > 0:  # left
        intent_direction = -1
    
    lines = []
    final_pts = []
    
    # Normalization factors for scoring
    max_dy_per_step = 4.0  # meters per 0.5s for progress
    max_lateral_change = 0.5  # meters of lateral change for smoothness
    
    for t in range(1, 7):  # steps 1..6 (0.5s to 3.0s)
        base = fut[t].copy()
        # Three candidates: center/base, right offset, left offset
        cands = np.array([
            [base[0], base[1]],                           # a (center/GT-like)
            [base[0] + lateral_offset, base[1]],          # b (right)
            [base[0] - lateral_offset, base[1]],          # c (left)
        ])
        
        # Score each candidate on all aspects
        scores = []
        for cand in cands:
            x, y = cand
            
            # Aspect 1: Safety (lane centering + proximity)
            safety = _clamp(1.0 - abs(x) / lane_half_width)
            
            # Aspect 2: Smoothness (avoid jerky lateral movements)
            lat_change = abs(x - prev[0])
            smoothness = _clamp(1.0 - (lat_change / max_lateral_change))
            
            # Aspect 3: Efficiency & intent alignment
            dy = y - prev[1]
            progress = _clamp(dy / max_dy_per_step)
            
            # Intent alignment bonus: favor right/left based on mission
            intent_bonus = 0
            if intent_direction != 0:
                # If turning right, favor rightward motion
                intent_match = (x - prev[0]) * intent_direction
                intent_bonus = _clamp(0.2 * intent_match)
            
            # Weighted final score
            s = 0.35 * safety + 0.25 * smoothness + 0.35 * progress + 0.05 * intent_bonus
            scores.append(_clamp(s))
        
        scores = np.array(scores)
        # Choose best candidate for final trajectory
        best_idx = int(scores.argmax())
        final_pts.append(cands[best_idx].tolist())
        
        # Format waypoint line
        a, b, c = cands
        sa, sb, sc = scores.tolist()
        line = (
            f"W{t}:["
            f"({a[0]:.2f},{a[1]:.2f})s={sa:.2f},"
            f"({b[0]:.2f},{b[1]:.2f})s={sb:.2f},"
            f"({c[0]:.2f},{c[1]:.2f})s={sc:.2f}"
            f"]"
        )
        lines.append(line)
        
        # Update prev for next step
        prev = cands[best_idx]
    
    # Format final trajectory
    final_str = "[" + ", ".join([f"({p[0]:.2f},{p[1]:.2f})" for p in final_pts]) + "]"
    
    # Join all ToT reasoning lines
    tot_str = "\n".join(lines)
    return tot_str, final_str


def generate_user_message(data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    user_message  = f"Here's some information you'll need:\n"
    
    data_dict = data[token]
    camera_types = [
            'CAM_FRONT',
        ]
    images_path = []
    for cam in camera_types:
        images_path.append(data_dict['cams'][cam]['data_path'].replace('/localdata_ssd/nuScenes', 'data/nuscenes', 1))

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0
        mission_goal = "FORWARD"
    user_message += f"Mission Goal: "
    user_message += f"{mission_goal}\n"

    """
    Traffic Rule
    """
    user_message += f"Traffic Rules: Avoid collision with other objects.\n- Always drive on drivable regions.\n- Avoid driving on occupied regions.\n"
    
    """
    Evaluation Rule
    """
    
    user_message += f"When evaluating waypoints, score based on safety (lane positioning), smoothness (gradual changes), and alignment with the {mission_goal} mission.\n"

    return user_message, images_path

# filepath: /media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Drive/create_data/prompt_message.py
def generate_assistant_message(data, token, traj_only=False):
    data_dict = data[token]
    fut = data_dict['gt_ego_fut_trajs']
    # Safety assertion (need at least 7 entries: index 0 current, 1..6 future)
    assert len(fut) >= 7, f"gt_ego_fut_trajs too short for token {token}: len={len(fut)}"

    # Explicit extraction 
    x1 = fut[1][0]; y1 = fut[1][1]
    x2 = fut[2][0]; y2 = fut[2][1]
    x3 = fut[3][0]; y3 = fut[3][1]
    x4 = fut[4][0]; y4 = fut[4][1]
    x5 = fut[5][0]; y5 = fut[5][1]
    x6 = fut[6][0]; y6 = fut[6][1]

    gt_final_str = (
        f"[({x1:.2f},{y1:.2f}), "
        f"({x2:.2f},{y2:.2f}), "
        f"({x3:.2f},{y3:.2f}), "
        f"({x4:.2f},{y4:.2f}), "
        f"({x5:.2f},{y5:.2f}), "
        f"({x6:.2f},{y6:.2f})]"
    )

    if traj_only:
        return gt_final_str
    else:
        tot_str, _ignored = generate_three_of_thoughts(data_dict)
        return tot_str + "\nFinal:" + gt_final_str