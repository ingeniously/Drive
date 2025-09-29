import json
from nuscenes.nuscenes import NuScenes

split_path = "./create_data/full_split.json"
gt_indices_path = "./MoVQGAN/gt_indices_sft.json"
dataroot = './LLaMA-Factory/data/nuscenes'

split = json.load(open(split_path))
gt_indices = json.load(open(gt_indices_path))
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)

def valid_token(token):
    try:
        next_token = nusc.get('sample', token)['next']
        return next_token in gt_indices and 'CAM_FRONT' in gt_indices[next_token]
    except:
        return False

for split_name in ["train", "val"]:
    split[split_name] = [t for t in split[split_name] if valid_token(t)]

with open(split_path, "w") as f:
    json.dump(split, f, indent=2)

print("Filtered split to only include tokens whose 'next' sample exists in gt_indices and has 'CAM_FRONT'.")