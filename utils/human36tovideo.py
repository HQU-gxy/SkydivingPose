import os
import cv2
from collections import defaultdict
import re

def create_video_from_images(image_folder, output_folder, action_name, camera_id, images, frame_rate=12):
    images = sorted(images)
    
    if not images:
        return

    # Get size of the images
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    video_name = f"{action_name}_{camera_id}.mp4"
    video_path = os.path.join(output_folder, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video {video_name} created successfully")

def main():
    directory = '/home/public/data/h36m/images/S1'
    output_directory = '/home/zlt/Documents/SkydivingPose/output/human36'
    os.makedirs(output_directory, exist_ok=True)

    file_names = os.listdir(directory)
    
    action_camera_map = defaultdict(lambda: defaultdict(list))

    pattern = re.compile(r'^(.*?)\.(\d+)_\d+\.jpg$')

    for file_name in file_names:
        match = pattern.match(file_name)
        if match:
            action_name = match.group(1)
            camera_id = match.group(2)
            action_camera_map[action_name][camera_id].append(file_name)

    for action_name, camera_ids in action_camera_map.items():
        for camera_id, images in camera_ids.items():
            print(camera_id)
            create_video_from_images(directory, output_directory, action_name, camera_id, images)

if __name__ == "__main__":
    main()
