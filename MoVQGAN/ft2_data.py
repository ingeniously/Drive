import os
import json
import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torchvision.transforms as T
from torch.nn.functional import mse_loss, l1_loss
from nuscenes.nuscenes import NuScenes
from movqgan import get_movqgan_model

def show_images(batch, file_path):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    image.save(file_path)


def prepare_image(img):
    """ Transform and normalize PIL Image to tensor. """
    transform = T.Compose([
            T.Resize((128, 192), interpolation=T.InterpolationMode.BICUBIC),
        ])
    pil_image = transform(img)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    
    return torch.from_numpy(np.transpose(arr, [2, 0, 1]))


import torch.multiprocessing as mp

def run_inference(rank, world_size, samples, cams, dataroot, output_dir, output_name):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    model = get_movqgan_model('270M', pretrained=True, device=device)
    model.eval()

    total_samples = len(samples)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    local_samples = samples[start_idx:end_idx]

    gt_indices = {}
    for rec in tqdm(local_samples, desc=f"Rank {rank} processing", position=rank):
        sample = {}
        for cam in cams:
            samp = nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(nusc.dataroot, samp['filename'])
            if not os.path.exists(imgname):
                print(f"Image not found: {imgname}")
                continue
            img = prepare_image(Image.open(imgname))
            with torch.no_grad():
                out = model(img.to(device).unsqueeze(0))
            sample[cam] = str(out.cpu().tolist())
        gt_indices[samp['sample_token']] = sample

    name = output_name.split('.')[0]
    output_json_path = os.path.join(output_dir, f"{name}_{rank}.json")
    with open(output_json_path, "w") as f:
        json.dump(gt_indices, f, indent=4)

def aggregate_results(world_size, output_dir, final_output_path, output_name):
    aggregated = {}
    name = output_name.split('.')[0]
    for rank in range(world_size):
        partial_json_path = os.path.join(output_dir, f"{name}_{rank}.json")
        if os.path.exists(partial_json_path):
            with open(partial_json_path, "r") as f:
                partial_data = json.load(f)
            aggregated.update(partial_data)
            os.remove(partial_json_path)
        else:
            print(f"Warning: {partial_json_path} not found")
    with open(final_output_path, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"All partial results merged and saved to {final_output_path}")

def main(dataroot, output_dir, output_name):
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
    samples = nusc.sample
    cams = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        #'CAM_BACK',
        #'CAM_BACK_LEFT',
        #'CAM_BACK_RIGHT',
    ]
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No available GPU on this machine.")
    os.makedirs(output_dir, exist_ok=True)
    final_output_path = os.path.join(output_dir, output_name)
    mp.spawn(
        run_inference,
        args=(world_size, samples, cams, dataroot, output_dir, output_name),
        nprocs=world_size,
        join=True
    )
    aggregate_results(world_size, output_dir, final_output_path, output_name)
    print(f"Processing completed. All results saved to {final_output_path}")

if __name__ == "__main__":
    dataroot = './LLaMA-Factory/data/nuscenes'
    output_dir = './MoVQGAN'
    output_name = 'gt_indices_ft2.json'
    main(dataroot, output_dir, output_name)


