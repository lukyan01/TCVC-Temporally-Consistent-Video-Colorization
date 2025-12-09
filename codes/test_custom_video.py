"""
TCVC Video Colorization - Enhanced Test Script
Supports both subfolder and single-folder video formats
Computes PSNR (if ground truth provided) and temporal consistency metrics
"""

import os
import os.path as osp
import glob
import logging
import argparse
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

import utils.util as util
import data.util as data_util
import models.archs.TCVC_IDC_arch as TCVC_IDC_arch

from compute_hist import *


def parse_args():
    parser = argparse.ArgumentParser(description='TCVC Video Colorization')

    # Required arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input folder with grayscale frames (can be direct folder or parent with subfolders)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output folder for colorized results')

    # Optional arguments
    parser.add_argument('--gt', type=str, default=None,
                        help='Ground truth RGB folder for PSNR calculation (optional)')
    parser.add_argument('--model', type=str, default='/workspace/baselines/tcvc/experiments/80000_G.pth',
                        help='Path to pretrained TCVC model (default: /workspace/baselines/tcvc/experiments/80000_G.pth)')
    parser.add_argument('--interval', type=int, default=8,
                        help='Interval length between keyframes (default: 8, options: 4/8/17)')
    parser.add_argument('--size', type=int, default=256,
                        help='Processing image size (default: 256)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--subfolder-mode', action='store_true',
                        help='Enable if input has video subfolders (INPUT/video1/frames.png)')

    return parser.parse_args()


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (0-255 range)"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_temporal_warp_error(frames):
    """Calculate temporal consistency using frame differences"""
    if len(frames) < 2:
        return 0.0

    total_diff = 0.0
    for i in range(len(frames) - 1):
        frame1 = frames[i].astype(np.float32)
        frame2 = frames[i + 1].astype(np.float32)
        diff = np.mean(np.abs(frame1 - frame2))
        total_diff += diff

    return total_diff / (len(frames) - 1)


def save_imglist(k, end_k, output_dir, img_list, logger, img_paths):
    """Save the colorized image list"""
    count = 0
    for i in range(k, end_k):
        imname = os.path.basename(img_paths[count])
        out_path = os.path.join(output_dir, imname)
        cv2.imwrite(out_path, img_list[count][:,:,::-1])
        count += 1


def process_video_folder(input_path, gt_path, output_path, model, device, args, logger):
    """Process a single video folder"""
    # Support multiple image formats
    img_list = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        img_list.extend(glob.glob(os.path.join(input_path, ext)))
    img_list = sorted(img_list)

    if not img_list:
        logger.warning(f"No images found in {input_path}, skipping...")
        return None

    logger.info(f"Found {len(img_list)} frames")

    # Get ground truth if provided
    gt_img_list = []
    if gt_path and os.path.exists(gt_path):
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            gt_img_list.extend(glob.glob(os.path.join(gt_path, ext)))
        gt_img_list = sorted(gt_img_list)
        logger.info(f"Found {len(gt_img_list)} ground truth frames")

    # Load images
    imgs = [data_util.read_img(None, img_list[i])/255. for i in range(len(img_list))]

    # Check if grayscale or RGB
    if imgs[0].shape[-1] == 3:
        rgb_flag = True
        logger.info("Input: RGB images (will extract L channel)")
    elif imgs[0].shape[-1] == 1:
        rgb_flag = False
        logger.info("Input: Grayscale images")
    else:
        logger.error(f'Unexpected image channels: {imgs[0].shape[-1]}')
        return None

    # Calculate keyframe indices
    keyframe_idx = list(range(0, len(imgs), args.interval + 1))
    if keyframe_idx[-1] == (len(imgs) - 1):
        keyframe_idx = keyframe_idx[:-1]

    logger.info(f"Keyframe indices: {keyframe_idx}")
    logger.info(f"Number of keyframes: {len(keyframe_idx)}")
    logger.info("Processing...")

    # Store output frames for metrics
    output_frames = []

    # Process video in chunks
    for chunk_idx, k in enumerate(keyframe_idx):
        logger.info(f"  Processing chunk {chunk_idx+1}/{len(keyframe_idx)} (frames {k} to {min(k+args.interval+2, len(imgs))})")

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

        # Convert to images
        out_rgb_img = [util.tensor2img(np.clip(out_rgb_origsize[i,...]*255., 0, 255), np.uint8)
                      for i in range(out_rgb_origsize.size(0))]

        # Store for temporal metrics
        output_frames.extend(out_rgb_img)

        # Save results
        save_imglist(k, k+len(out_rgb_img), output_path, out_rgb_img, logger, img_paths)

    # Calculate PSNR if ground truth provided
    psnr_values = []
    if gt_img_list and len(gt_img_list) == len(output_frames):
        logger.info("Calculating PSNR...")
        for i, (out_frame, gt_path) in enumerate(zip(output_frames, gt_img_list)):
            gt_frame = cv2.imread(gt_path)
            if gt_frame is not None:
                psnr = calculate_psnr(out_frame[:,:,::-1], gt_frame)  # BGR to RGB
                psnr_values.append(psnr)
                if i < 5 or i % 10 == 0:  # Log first 5 and every 10th
                    logger.info(f"  Frame {i+1:04d}: PSNR = {psnr:.2f} dB")

    # Calculate temporal consistency
    temporal_error = calculate_temporal_warp_error(output_frames)

    return {
        'num_frames': len(output_frames),
        'psnr_values': psnr_values,
        'avg_psnr': np.mean(psnr_values) if psnr_values else None,
        'temporal_error': temporal_error
    }


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Setup model
    model = TCVC_IDC_arch.TCVC_IDC(nf=64, N_RBs=3, key_net="sig17", dataset="DAVIS4")

    # Setup folders
    util.mkdirs(args.output)
    util.setup_logger(
        "base", args.output, "test", level=logging.INFO, screen=True, tofile=True
    )
    logger = logging.getLogger("base")

    # Log info
    logger.info("="*70)
    logger.info("TCVC Video Colorization")
    logger.info("="*70)
    logger.info(f"Input folder: {args.input}")
    logger.info(f"Output folder: {args.output}")
    logger.info(f"Ground truth: {args.gt if args.gt else 'None'}")
    logger.info(f"Model path: {args.model}")
    logger.info(f"Interval length: {args.interval}")
    logger.info(f"Process size: {args.size}")
    logger.info(f"GPU ID: {args.gpu}")
    logger.info(f"Subfolder mode: {args.subfolder_mode}")
    logger.info("="*70)

    # Load model
    logger.info("Loading model...")
    model.load_state_dict(torch.load(args.model), strict=True)
    model.eval()
    model = model.to(device)
    logger.info("Model loaded successfully!")

    # Determine mode: single video or multiple videos
    if args.subfolder_mode:
        # Multiple videos in subfolders
        video_list = sorted(os.listdir(args.input))
        video_list = [v for v in video_list if os.path.isdir(os.path.join(args.input, v))]

        if not video_list:
            logger.error(f"No video folders found in {args.input}")
            return

        logger.info(f"Found {len(video_list)} video(s) to process: {video_list}")

    # Process videos
    all_results = []

    if args.subfolder_mode:
        # Process multiple videos in subfolders
        for video_idx, video in enumerate(video_list):
            logger.info("")
            logger.info("="*70)
            logger.info(f"Processing video {video_idx+1}/{len(video_list)}: '{video}'")
            logger.info("="*70)

            input_path = os.path.join(args.input, video)
            gt_path = os.path.join(args.gt, video) if args.gt else None
            output_path = os.path.join(args.output, video)
            util.mkdirs(output_path)

            result = process_video_folder(input_path, gt_path, output_path, model, device, args, logger)
            if result:
                result['name'] = video
                all_results.append(result)
                logger.info(f"Finished '{video}' - saved to {output_path}")
    else:
        # Process single video directly in input folder
        logger.info("")
        logger.info("="*70)
        logger.info("Processing video...")
        logger.info("="*70)

        util.mkdirs(args.output)
        result = process_video_folder(args.input, args.gt, args.output, model, device, args, logger)
        if result:
            result['name'] = 'video'
            all_results.append(result)
            logger.info(f"Finished - saved to {args.output}")

    # Calculate temporal consistency metrics using compute_hist
    logger.info("")
    logger.info("="*70)
    logger.info("Calculating color distribution consistency metrics...")
    logger.info("="*70)

    try:
        dilation = [1,2,4]
        weight = [1/3, 1/3, 1/3]
        JS_b_mean_list, JS_g_mean_list, JS_r_mean_list, JS_b_dict, JS_g_dict, JS_r_dict, CDC = \
            calculate_folders_multiple(args.output, "Real", dilation=dilation, weight=weight)

        logger.info(f"JS_b_mean: {np.mean(JS_b_mean_list):.6f}")
        logger.info(f"JS_g_mean: {np.mean(JS_g_mean_list):.6f}")
        logger.info(f"JS_r_mean: {np.mean(JS_r_mean_list):.6f}")
        logger.info(f"CDC (Color Distribution Consistency): {CDC:.6f}")
    except Exception as e:
        logger.warning(f"Could not calculate CDC metrics: {e}")

    # Print summary
    logger.info("")
    logger.info("="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)

    for result in all_results:
        logger.info(f"\nVideo: {result['name']}")
        logger.info(f"  Frames processed: {result['num_frames']}")
        logger.info(f"  Temporal error (frame diff): {result['temporal_error']:.4f}")
        if result['avg_psnr'] is not None:
            logger.info(f"  Average PSNR: {result['avg_psnr']:.2f} dB")
            logger.info(f"  Min PSNR: {np.min(result['psnr_values']):.2f} dB")
            logger.info(f"  Max PSNR: {np.max(result['psnr_values']):.2f} dB")

    logger.info("")
    logger.info("="*70)
    logger.info("All processing complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
