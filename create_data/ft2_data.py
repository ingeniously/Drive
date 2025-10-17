import pickle
import re
import json
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import generate_user_message, generate_assistant_message, generate_three_of_thoughts
import os

# Update system prompt to describe the ToT format
system = (
  "You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. "
    "You're at point (0,0). Units: meters. Based on the provided particulars, output the CAM_FRONT image at 0.5 seconds in the future, "
    "evaluate multiple trajectory options with scores for safety, smoothness, and efficiency, then select the best path.\n"
    "When scoring waypoint options, consider:\n"
    "- Safety: Higher scores for positions closer to lanes center\n"
    "- Smoothness: Higher scores for appropriate movements\n"
    "- Intent: Slight bonus when movement aligns with mission goal\n"
    "Answer format:\n"
    "<future_image_tokens>\n"
    "W1:[(x1_a,y1_a)s=0.0000,(x1_b,y1_b)s=0.0000,(x1_c,y1_c)s=0.0000]\n"
    "W2:[(x2_a,y2_a)s=0.0000,(x2_b,y2_b)s=0.0000,(x2_c,y2_c)s=0.00]\n"
    "W3:[...] through W6:[...]\n"
    "Final:[(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]"
)

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('./create_data/full_split.json', 'r'))
tokens = split[args.split]

num_train_samples = len(tokens)
train_ratio = 1

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = False  # We want ToT reasoning, not just trajectory

dataroot = './LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
sft_indices = json.load(open('./MoVQGAN/gt_indices_ft2.json'))
train_messages = []

for token_i, token in enumerate(tokens):
    if token_i >= train_ratio * num_train_samples:
        break
        
    user_message, images_path = generate_user_message(data, token)
    
    # Generate assistant message with ToT reasoning
    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)

    # Get visual tokens for future image
    try:
        next_token = nusc.get('sample', token)['next']
        next_img_token = sft_indices[next_token]['CAM_FRONT']
        next_img_token = str(next_img_token).replace(" ", "")
        numbers = next_img_token.strip('[]').split(',')
        next_img_token = ''.join([f'<|{num}|>' for num in numbers])
    except:
        continue

    # Token accounting
    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))

    train_message = {
        "id": token,
        "images": images_path,
        "system": system,
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Here is the current image from the car: 'CAM_FRONT': <image>\n,"
                    + user_message
                    + "Based on the provided particulars, output:\n"
                    "1. The visual tokens for the future image at 0.5 seconds\n"
                    "2. For each waypoint (W1-W6), three options with safety, smoothness, and efficiency scores\n"
                    "3. A final trajectory with the best waypoints\n"
                ),
            },
            {
                "from": "gpt",
                "value": next_img_token + "\n" + assitant_message + " <|endoftext|><|im_end|>"
            },
        ],
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")

with open(f"./LLaMA-Factory/data/{args.split}_ft2_cot_tot_motion.json", "w") as f:
    json.dump(train_messages, f, indent=4)