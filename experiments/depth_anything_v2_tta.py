#!/usr/bin/env python3
"""
Depth Anything V2 ONNX Inference with TTA (Test Time Augmentation)
Performs horizontal flip TTA and saves all intermediate results
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path


def normalize_depth(depth_map):
    """Normalize depth map to 0-255 range for visualization"""
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 1e-8:
        normalized = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
    else:
        normalized = np.zeros_like(depth_map)
    return normalized.astype(np.uint8)


def preprocess_image(image, input_size=(518, 518)):
    """Preprocess image for Depth Anything V2 model"""
    # Resize image
    h, w = image.shape[:2]
    resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std

    # Convert to NCHW format
    input_tensor = np.transpose(normalized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

    return input_tensor, (h, w)


def run_inference(session, input_tensor):
    """Run inference using ONNX Runtime"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: input_tensor})
    depth = outputs[0]

    return depth


def postprocess_depth(depth, original_size):
    """Postprocess depth map to original image size"""
    # Remove batch dimension
    if depth.ndim == 4:
        depth = depth.squeeze(0)
    if depth.ndim == 3:
        depth = depth.squeeze(0)

    # Resize to original size
    h, w = original_size
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth_resized


def compute_difference_map(depth1, depth2):
    """Compute absolute difference between two depth maps"""
    diff = np.abs(depth1 - depth2)
    return diff


def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 ONNX Inference with TTA')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='./extracted_frames', help='Output directory')
    parser.add_argument('--input_size', type=int, default=518, help='Model input size')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ONNX model
    print(f"Loading ONNX model from: {args.model}")
    session = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"Using provider: {session.get_providers()[0]}")

    # Load image
    print(f"Loading image from: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image}")

    original_size = (image.shape[0], image.shape[1])
    print(f"Image size: {original_size[1]}x{original_size[0]}")

    # Save original image
    output_original = output_dir / "1_original.png"
    cv2.imwrite(str(output_original), image)
    print(f"Saved: {output_original}")

    # Process original image
    print("\nProcessing original image...")
    input_tensor, _ = preprocess_image(image, input_size=(args.input_size, args.input_size))
    depth_original = run_inference(session, input_tensor)
    depth_original_resized = postprocess_depth(depth_original, original_size)

    # Save original depth map
    depth_original_vis = normalize_depth(depth_original_resized)
    depth_original_colored = cv2.applyColorMap(depth_original_vis, cv2.COLORMAP_INFERNO)
    output_depth_original = output_dir / "2_depth_original.png"
    cv2.imwrite(str(output_depth_original), depth_original_colored)
    print(f"Saved: {output_depth_original}")

    # Horizontal flip
    print("\nProcessing flipped image...")
    image_flipped = cv2.flip(image, 1)  # 1 for horizontal flip
    output_flipped = output_dir / "3_flipped.png"
    cv2.imwrite(str(output_flipped), image_flipped)
    print(f"Saved: {output_flipped}")

    # Process flipped image
    input_tensor_flipped, _ = preprocess_image(image_flipped, input_size=(args.input_size, args.input_size))
    depth_flipped = run_inference(session, input_tensor_flipped)
    depth_flipped_resized = postprocess_depth(depth_flipped, original_size)

    # Save flipped depth map
    depth_flipped_vis = normalize_depth(depth_flipped_resized)
    depth_flipped_colored = cv2.applyColorMap(depth_flipped_vis, cv2.COLORMAP_INFERNO)
    output_depth_flipped = output_dir / "4_depth_flipped.png"
    cv2.imwrite(str(output_depth_flipped), depth_flipped_colored)
    print(f"Saved: {output_depth_flipped}")

    # Flip depth map back
    print("\nFlipping depth map back...")
    depth_flipped_back = cv2.flip(depth_flipped_resized, 1)

    # Save flipped-back depth map
    depth_flipped_back_vis = normalize_depth(depth_flipped_back)
    depth_flipped_back_colored = cv2.applyColorMap(depth_flipped_back_vis, cv2.COLORMAP_INFERNO)
    output_depth_flipped_back = output_dir / "5_depth_flipped_back.png"
    cv2.imwrite(str(output_depth_flipped_back), depth_flipped_back_colored)
    print(f"Saved: {output_depth_flipped_back}")

    # Compute difference maps
    print("\nComputing difference maps...")

    # Difference between original and flipped-back
    diff_map = compute_difference_map(depth_original_resized, depth_flipped_back)
    diff_map_vis = normalize_depth(diff_map)
    diff_map_colored = cv2.applyColorMap(diff_map_vis, cv2.COLORMAP_JET)
    output_diff = output_dir / "6_difference_original_vs_flipped_back.png"
    cv2.imwrite(str(output_diff), diff_map_colored)
    print(f"Saved: {output_diff}")

    # Alternative difference visualization (grayscale)
    output_diff_gray = output_dir / "7_difference_grayscale.png"
    cv2.imwrite(str(output_diff_gray), diff_map_vis)
    print(f"Saved: {output_diff_gray}")

    # Compute statistics
    print("\n=== Statistics ===")
    print(f"Original depth - min: {depth_original_resized.min():.4f}, max: {depth_original_resized.max():.4f}, mean: {depth_original_resized.mean():.4f}")
    print(f"Flipped-back depth - min: {depth_flipped_back.min():.4f}, max: {depth_flipped_back.max():.4f}, mean: {depth_flipped_back.mean():.4f}")
    print(f"Difference - min: {diff_map.min():.4f}, max: {diff_map.max():.4f}, mean: {diff_map.mean():.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(depth_original_resized - depth_flipped_back)):.4f}")

    # TTA averaged depth (optional)
    print("\nComputing TTA averaged depth...")
    depth_tta_avg = (depth_original_resized + depth_flipped_back) / 2.0
    depth_tta_avg_vis = normalize_depth(depth_tta_avg)
    depth_tta_avg_colored = cv2.applyColorMap(depth_tta_avg_vis, cv2.COLORMAP_INFERNO)
    output_tta_avg = output_dir / "8_depth_tta_averaged.png"
    cv2.imwrite(str(output_tta_avg), depth_tta_avg_colored)
    print(f"Saved: {output_tta_avg}")

    print("\n=== All outputs saved successfully! ===")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
