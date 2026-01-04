#!/usr/bin/env python3
"""
Extract frames from ROS bag file
Extracts 20 frames from specified image topic and saves them as PNG files
"""

import rosbag
import cv2
from cv_bridge import CvBridge
import argparse
from pathlib import Path
import numpy as np


def extract_frames(bag_path, topic, output_dir, num_frames=20):
    """
    Extract frames from ROS bag file

    Args:
        bag_path: Path to ROS bag file
        topic: Image topic to extract from
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()

    print(f"Opening bag file: {bag_path}")
    print(f"Extracting {num_frames} frames from topic: {topic}")

    # Open bag file
    bag = rosbag.Bag(bag_path, 'r')

    # Get total number of messages in the topic
    total_messages = bag.get_message_count(topic_filters=topic)
    print(f"Total messages in topic '{topic}': {total_messages}")

    if total_messages == 0:
        print(f"Error: No messages found in topic '{topic}'")
        print("\nAvailable topics:")
        topic_info = bag.get_type_and_topic_info()
        for t, info in topic_info.topics.items():
            print(f"  - {t} ({info.msg_type}): {info.message_count} messages")
        bag.close()
        return

    # Calculate stride to sample frames evenly
    if total_messages <= num_frames:
        stride = 1
        frames_to_extract = total_messages
        print(f"Warning: Only {total_messages} frames available, extracting all of them")
    else:
        stride = total_messages // num_frames
        frames_to_extract = num_frames

    print(f"Sampling every {stride} frame(s)")

    # Extract frames
    frame_count = 0
    msg_index = 0

    for topic_name, msg, t in bag.read_messages(topics=[topic]):
        if msg_index % stride == 0 and frame_count < frames_to_extract:
            try:
                # Convert ROS Image message to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Save frame
                timestamp = msg.header.stamp.to_sec()
                frame_filename = output_dir / f"frame_{frame_count:04d}_t{timestamp:.6f}.png"
                cv2.imwrite(str(frame_filename), cv_image)

                print(f"Saved frame {frame_count + 1}/{frames_to_extract}: {frame_filename.name} "
                      f"(size: {cv_image.shape[1]}x{cv_image.shape[0]})")

                frame_count += 1

            except Exception as e:
                print(f"Error processing message {msg_index}: {e}")

        msg_index += 1

        if frame_count >= frames_to_extract:
            break

    bag.close()

    print(f"\n=== Extraction complete ===")
    print(f"Extracted {frame_count} frames to: {output_dir.absolute()}")

    # Save frame list
    frame_list_file = output_dir / "frame_list.txt"
    with open(frame_list_file, 'w') as f:
        for i in range(frame_count):
            frames = sorted(output_dir.glob("frame_*.png"))
            for frame in frames:
                f.write(f"{frame.name}\n")
    print(f"Frame list saved to: {frame_list_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract frames from ROS bag file')
    parser.add_argument('--bag', type=str, required=True, help='Path to ROS bag file')
    parser.add_argument('--topic', type=str, default='/cam0/image_raw',
                        help='Image topic to extract (default: /cam0/image_raw)')
    parser.add_argument('--output_dir', type=str, default='./extracted_frames',
                        help='Output directory for extracted frames (default: ./extracted_frames)')
    parser.add_argument('--num_frames', type=int, default=20,
                        help='Number of frames to extract (default: 20)')

    args = parser.parse_args()

    # Check if bag file exists
    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"Error: Bag file not found: {args.bag}")
        return

    extract_frames(args.bag, args.topic, args.output_dir, args.num_frames)


if __name__ == "__main__":
    main()
