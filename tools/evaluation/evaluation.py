import torch
from torch import Tensor
from tqdm import tqdm
import pickle
import json
from pathlib import Path
import os
import re
import ast
import argparse
import logging

logging.basicConfig(level=logging.INFO)

FINAL_COORDS_RE = re.compile(
    r'Final:\s*\[(.*?)\]',
    re.DOTALL
)

PAIR_RE = re.compile(
    r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
)

def extract_final_pairs(pred_text, expected_pairs=6):
    """
    Extract only the Final:[(x,y), ...] block.
    Returns list of (x,y) floats or None if not found / count mismatch.
    """
    m = FINAL_COORDS_RE.search(pred_text)
    if not m:
        return None, "No 'Final:' block"
    inner = m.group(1)
    pairs = PAIR_RE.findall(inner)
    if len(pairs) != expected_pairs:
        return None, f"Expected {expected_pairs} pairs, found {len(pairs)}"
    return [(float(x), float(y)) for x, y in pairs], None


def planning_evaluation(pred_trajs_dict, config):
    future_second = 3
    ts = future_second * 2
    device = torch.device('cpu')

    if config.metric=="uniad":
        from metric_uniad import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Test_drive_tot_vla/FSDrive2/tools/data/metrics/uniad_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
    elif config.metric=="stp3":
        from metric_stp3 import PlanningMetric
        with open(Path(os.path.join(config.gt_folder, '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Test_drive_tot_vla/FSDrive2/tools/data/metrics/stp3_gt_seg.pkl')),'rb') as f:
            gt_occ_map = pickle.load(f)
        for token in gt_occ_map.keys():
            if not isinstance(gt_occ_map[token], torch.Tensor):
                gt_occ_map[token] = torch.tensor(gt_occ_map[token])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-1])
            gt_occ_map[token] = torch.flip(gt_occ_map[token], [-2])
    else:
        raise ValueError(f"Invalid metric: {config.metric}")
    
    metric_planning_val = PlanningMetric(ts).to(device)     

    with open(Path(os.path.join(config.gt_folder, '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Test_drive_tot_vla/FSDrive2/tools/data/metrics/gt_traj.pkl')),'rb') as f:
        gt_trajs_dict = pickle.load(f)

    with open(Path(os.path.join(config.gt_folder, '/media/seonho/34B6BDA8B6BD6ACE/AAAI/project/Test_drive_tot_vla/FSDrive2/tools/data/metrics/gt_traj_mask.pkl')),'rb') as f:
        gt_trajs_mask_dict = pickle.load(f)

    for index, token in enumerate(tqdm(gt_trajs_dict.keys())):

        gt_trajectory =  torch.tensor(gt_trajs_dict[token])
        gt_trajectory = gt_trajectory.to(device)

        gt_traj_mask = torch.tensor(gt_trajs_mask_dict[token])
        gt_traj_mask = gt_traj_mask.to(device)
        
        pred_text = pred_trajs_dict.get(token, "")
        final_pairs, err = extract_final_pairs(pred_text, expected_pairs=6)
        if final_pairs is None:
            logging.warning(f"[Skip] token={token} reason={err}")
            continue

        # final_pairs: list of 6 (x,y)
        # Convert to tensor shape (1, 6, 2) or flatten depending on what mask expects
        output_trajs = torch.tensor(final_pairs, dtype=torch.float32).unsqueeze(0)  # (1,6,2)

        # Now align with mask shape; examine gt_traj_mask shape first:
        # If gt_traj_mask is (1, 12) (meaning flattened x,y), then:
        if len(gt_traj_mask.shape) == 2 and gt_traj_mask.shape[1] == 12:
            output_trajs = output_trajs.reshape(1, 12)
        elif len(gt_traj_mask.shape) == 3 and gt_traj_mask.shape[1] == 6 and gt_traj_mask.shape[2] == 2:
            # matches directly, keep (1,6,2)
            pass
        else:
            logging.warning(f"Unexpected mask shape {gt_traj_mask.shape} for token={token}; skipping.")
            continue
        

        occupancy: Tensor = gt_occ_map[token]
        occupancy = occupancy.to(device)

        if output_trajs.shape[1] % 2: # in case the current timestep is inculded
            output_trajs = output_trajs[:, 1:]

        if occupancy.shape[1] % 2: # in case the current timestep is inculded
            occupancy = occupancy[:, 1:]
        
        if gt_trajectory.shape[1] % 2: # in case the current timestep is inculded
            gt_trajectory = gt_trajectory[:, 1:]

        if gt_traj_mask.shape[1] % 2:  # in case the current timestep is inculded
            gt_traj_mask = gt_traj_mask[:, 1:]
        
        metric_planning_val(output_trajs[:, :ts], gt_trajectory[:, :ts], occupancy[:, :ts], token, gt_traj_mask)
          
    results = {}
    scores = metric_planning_val.compute()
    for i in range(future_second):
        for key, value in scores.items():
            results['plan_'+key+'_{}s'.format(i+1)]=value[:(i+1)*2].mean()

    headers = ["Method", "L2 (m)", "Collision (%)"]
    sub_headers = ["1s", "2s", "3s", "Avg."]
    if config.metric=="uniad":
        method = (config.method, "{:.2f}".format(scores["L2"][1]), "{:.2f}".format(scores["L2"][3]), "{:.2f}".format(scores["L2"][5]),\
                "{:.2f}".format((scores["L2"][1]+ scores["L2"][3]+ scores["L2"][5]) / 3.), \
                "{:.2f}".format(scores["obj_box_col"][1]*100), \
                "{:.2f}".format(scores["obj_box_col"][3]*100), \
                "{:.2f}".format(scores["obj_box_col"][5]*100), \
                "{:.2f}".format(100*(scores["obj_box_col"][1]+ scores["obj_box_col"][3]+ scores["obj_box_col"][5]) / 3.))
        print("{:<15} {:<20} {:<20}".format(*headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    elif config.metric=="stp3":
        method = (config.method, "{:.2f}".format(results["plan_L2_1s"]), "{:.2f}".format(results["plan_L2_2s"]), "{:.2f}".format(results["plan_L2_3s"]), \
                    "{:.2f}".format((results["plan_L2_1s"]+results["plan_L2_2s"]+results["plan_L2_3s"])/3.), \
                    "{:.2f}".format(results["plan_obj_box_col_1s"]*100), "{:.2f}".format(results["plan_obj_box_col_2s"]*100), "{:.2f}".format(results["plan_obj_box_col_3s"]*100), \
                        "{:.2f}".format(((results["plan_obj_box_col_1s"] + results["plan_obj_box_col_2s"] + results["plan_obj_box_col_3s"])/3)*100))
        print("{:<15} {:<20} {:<20}".format(*headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format("", *sub_headers, *sub_headers))
        print("{:<15} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}".format(*method))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of planning')
    parser.add_argument('--method', type=str, help='name of the method being evaluated, used for table print', default='Drive')
    parser.add_argument('--result_file', type=str, help='path to the result file from match.py', default='eval_traj.json')
    parser.add_argument('--metric', type=str, default='uniad', help='metric to evaluate, either uniad or stp3')
    parser.add_argument('--gt_folder', type=str, default='tools/data/metrics')
    config = parser.parse_args()

    result_file = Path(config.result_file)
    pred_trajs_dict = json.load(open(config.result_file, 'r'))
    planning_evaluation(pred_trajs_dict, config)