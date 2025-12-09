"""
TCVC Inference Script - Simplified for Pipeline Integration

This script performs core TCVC colorization without extra metrics computation.
Metrics (PSNR, temporal consistency) are handled by the main pipeline.
"""

import os
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import utils.util as util
import data.util as data_util
import models.archs.TCVC_IDC_arch as TCVC_IDC_arch


def parse_args():
    parser = argparse.ArgumentParser(description='TCVC Video Colorization - Inference Only')

    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder with grayscale frames')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for colorized results')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained TCVC model')

    # Optional arguments
    parser.add_argument('--interval', type=int, default=8,
                        help='Interval length between keyframes (default: 8, options: 4/8/17)')
    parser.add_argument('--size', type=int, default=256,
                        help='Processing image size (default: 256)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (default: 0)')

    return parser.parse_args()


def process_frames(input_dir, output_dir, model, args):
    """
    Process grayscale frames with TCVC model.

    Args:
        input_dir: Directory containing grayscale input frames
        output_dir: Directory to save colorized frames
        model: Loaded TCVC model
        args: Command line arguments

    Returns:
        dict: Processing statistics
    """
    # Find input frames
    img_list = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        img_list.extend(glob.glob(os.path.join(input_dir, ext)))
    img_list = sorted(img_list)

    if not img_list:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Found {len(img_list)} frames to process")

    # Create output directory
    util.mkdirs(output_dir)

    # Load images
    imgs = [data_util.read_img(None, img_list[i])/255. for i in range(len(img_list))]

    # Check if grayscale or RGB
    if imgs[0].shape[-1] == 3:
        rgb_flag = True
        print("Input: RGB images (will extract L channel)")
    elif imgs[0].shape[-1] == 1:
        rgb_flag = False
        print("Input: Grayscale images")
    else:
        raise ValueError(f'Unexpected image channels: {imgs[0].shape[-1]}')

    # Calculate keyframe indices
    keyframe_idx = list(range(0, len(imgs), args.interval + 1))
    if keyframe_idx[-1] == (len(imgs) - 1):
        keyframe_idx = keyframe_idx[:-1]

    print(f"Keyframe indices: {keyframe_idx}")
    print(f"Processing {len(keyframe_idx)} chunks...")

    # Process video in chunks
    frames_processed = 0

    for chunk_idx, k in enumerate(keyframe_idx):
        print(f"  Chunk {chunk_idx+1}/{len(keyframe_idx)} (frames {k} to {min(k+args.interval+2, len(imgs))})")

        img_paths = img_list[k:k+args.interval+2]
        img_in = imgs[k:k+args.interval+2]
        img_in = np.stack(img_in, 0)  # [N, H, W, C]
        img_tensor = torch.from_numpy(img_in.transpose(0,3,1,2)).float()

        # Convert to LAB or normalize grayscale
        if rgb_flag:
            img_lab_tensor = data_util.rgb2lab(img_tensor)
            img_l_tensor = img_lab_tensor[:,:1,:,:]
        else:
            img_l_tensor = img_tensor - 0.5

        # Resize to processing size
        img_l_rs_tensor = F.interpolate(img_l_tensor, size=[args.size, args.size], mode="bilinear")
        img_l_rs_tensor_list = [img_l_rs_tensor[i:i+1,...].cuda() for i in range(img_l_rs_tensor.shape[0])]

        # Inference
        with torch.no_grad():
            out_ab, _, _, _, _ = model(img_l_rs_tensor_list)

        out_ab = out_ab.detach().cpu()[0,...]

        # Resize back to original size
        N, C, H, W = img_tensor.size()
        out_a_rs = F.interpolate(out_ab[:,:1,:,:], size=[H, W], mode="bilinear")
        out_b_rs = F.interpolate(out_ab[:,1:2,:,:], size=[H, W], mode="bilinear")

        # Combine L channel with predicted AB
        out_lab_origsize = torch.cat((img_l_tensor, out_a_rs, out_b_rs), 1)
        out_rgb_origsize = data_util.lab2rgb(out_lab_origsize)

        # Convert to images and save
        for i in range(out_rgb_origsize.size(0)):
            img_rgb = util.tensor2img(np.clip(out_rgb_origsize[i,...]*255., 0, 255), np.uint8)

            # Get output filename
            imname = os.path.basename(img_paths[i])
            out_path = os.path.join(output_dir, imname)

            # Save (convert RGB to BGR for cv2)
            cv2.imwrite(out_path, img_rgb[:,:,::-1])
            frames_processed += 1

    print(f"\nProcessed {frames_processed} frames successfully")

    return {
        'frames_processed': frames_processed,
        'num_chunks': len(keyframe_idx),
        'interval': args.interval,
        'size': args.size,
    }


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("="*70)
    print("TCVC Video Colorization - Inference")
    print("="*70)
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Interval: {args.interval}")
    print(f"Process size: {args.size}")
    print(f"GPU ID: {args.gpu}")
    print("="*70)

    # Setup model
    print("\nLoading TCVC model...")
    model = TCVC_IDC_arch.TCVC_IDC(nf=64, N_RBs=3, key_net="sig17", dataset="DAVIS4")
    model.load_state_dict(torch.load(args.model), strict=True)
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")

    # Process frames
    print("\nProcessing frames...")
    result = process_frames(args.input, args.output, model, args)

    # Summary
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Frames processed: {result['frames_processed']}")
    print(f"Chunks processed: {result['num_chunks']}")
    print(f"Output saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
