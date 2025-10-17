#!/bin/bash

# This script downloads all 10 parts of the nuScenes v1.0 trainval set,
# plus the metadata and map files.
# IMPORTANT: You must get the actual URLs by logging into www.nuscenes.org

echo "Starting nuScenes full dataset download..."

# --- Metadata, Samples, and Maps ---
wget -c "https://www.nuscenes.org/data/v1.0-trainval_meta.tgz"
wget -c "https://www.nuscenes.org/data/v1.0-test_meta.tgz"
wget -c "https://www.nuscenes.org/data/maps.tgz"

# --- Trainval Set Blobs (10 parts) ---
# Replace these with the actual, full URLs from your account!
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval03_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval05_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval06_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval07_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz"
wget -c "https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz"

echo "Download complete! 🎉"