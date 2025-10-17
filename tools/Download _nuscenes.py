#!/usr/bin/env python3

import os
import sys
import subprocess
import requests
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import shutil


def check_requirements():
    """Check if required tools are installed."""
    required_tools = ['wget', 'tar', 'unzip']
    missing_tools = []
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"Error: Missing required tools: {', '.join(missing_tools)}")
        print("Please install them before continuing.")
        sys.exit(1)


def create_directory(directory):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory ready: {directory}")
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")
        return False


def check_disk_space(directory, required_gb=200):
    """Check if there's enough disk space."""
    try:
        # Get disk usage statistics
        total, used, free = shutil.disk_usage(directory)
        free_gb = free / (1024 ** 3)  # Convert to GB
        
        if free_gb < required_gb:
            print(f"Warning: You may not have enough disk space.")
            print(f"Available: {free_gb:.1f} GB, Recommended: {required_gb} GB")
            proceed = input("Do you want to continue anyway? (y/n): ")
            return proceed.lower() == 'y'
        return True
    except Exception as e:
        print(f"Error checking disk space: {str(e)}")
        proceed = input("Unable to check disk space. Continue anyway? (y/n): ")
        return proceed.lower() == 'y'


def download_file(url, output_path):
    """Download a file using wget with progress display."""
    try:
        # Using wget through subprocess for better progress display
        cmd = ['wget', '-c', url, '-O', output_path]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def extract_archive(file_path):
    """Extract tar.gz or zip archive."""
    try:
        print(f"Extracting: {file_path}")
        if file_path.endswith('.tgz') or file_path.endswith('.tar.gz'):
            cmd = ['tar', '-zxf', file_path]
            subprocess.run(cmd, check=True)
        elif file_path.endswith('.zip'):
            cmd = ['unzip', file_path]
            subprocess.run(cmd, check=True)
        else:
            print(f"Unsupported archive format: {file_path}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {file_path}: {str(e)}")
        return False


def main():
    """Main function to download and extract nuScenes dataset."""
    parser = argparse.ArgumentParser(description="Download and extract nuScenes dataset")
    parser.add_argument("--dest", default="nuscenes_tgz", help="Destination directory for the dataset")
    args = parser.parse_args()
    
    # Check if required tools are installed
    check_requirements()
    
    # Create dataset directory
    dataset_dir = Path(args.dest)
    if not create_directory(dataset_dir):
        sys.exit(1)
    
    # Check disk space
    if not check_disk_space(dataset_dir):
        sys.exit(1)
    
    # Change to dataset directory
    original_dir = os.getcwd()
    os.chdir(dataset_dir)
    
    print("Starting nuScenes dataset download...")
    
    # Define files to download with URLs
    # Replace 'PASTE_URL_HERE' with actual URLs from nuScenes website
    files_to_download = {
        # Metadata files
        "v1.0-trainval_meta.tgz": "PASTE_URL_HERE",
        "v1.0-test_meta.tgz": "PASTE_URL_HERE",
        "v1.0-nuscenes_lidarseg_splits_trainval.json": "PASTE_URL_HERE",
        "v1.0-lidarseg.tgz": "PASTE_URL_HERE",
        "v1.0-panoptic.tgz": "PASTE_URL_HERE",
        "nuimages-v1.0.tgz": "PASTE_URL_HERE",
        "v1.0-can_bus.tgz": "PASTE_URL_HERE",
        
        # Camera data
        "v1.0-trainval01_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval02_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval03_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval04_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval05_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval06_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval07_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval08_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval09_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-trainval10_blobs_camera.tgz": "PASTE_URL_HERE",
        "v1.0-test_blobs_camera.tgz": "PASTE_URL_HERE",
    }
    
    # Download each file
    downloaded_files = []
    for filename, url in files_to_download.items():
        print(f"\n{'-'*60}")
        print(f"Downloading: {filename}")
        print(f"{'-'*60}")
        
        if url == "PASTE_URL_HERE":
            print("URL not provided. Please replace with actual URL from nuScenes website.")
            proceed = input("Skip this file? (y/n): ")
            if proceed.lower() != 'y':
                break
            continue
        
        if download_file(url, filename):
            downloaded_files.append(filename)
    
    # Extract downloaded files
    print("\nDownload complete. Starting extraction...")
    for filename in downloaded_files:
        extract_archive(filename)
    
    # Return to original directory
    os.chdir(original_dir)
    
    print("\nExtraction complete.")
    print(f"The nuScenes dataset is now ready in the '{dataset_dir}' directory.")
    print("Please verify the contents against the official file list.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)