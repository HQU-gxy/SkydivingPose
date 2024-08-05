import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_skeleton_data(npz_path):
    data = np.load(npz_path)['reconstruction'][0]  # Assuming the first sequence
    return data[0]  # Return the first frame skeleton data

def plot_skeleton_on_image(ax, image, skeleton):
    for point in skeleton:
        ax.scatter(point[0], point[1], color='red', s=10)  # Red points for joints
    ax.imshow(image)
    ax.axis('off')

def main():
    npz_folder = '/home/zlt/Documents/SkydivingPose/output/test-001'
    video_folder = '/home/zlt/Documents/SkydivingPose/sample/test-001'
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
    fig, axs = plt.subplots(2, len(npz_files), figsize=(len(npz_files) * 5, 10))
    
    for i, npz_file in enumerate(sorted(npz_files)):
        video_file = npz_file.replace('_keypoints_2d.npz', '.mp4')
        npz_path = os.path.join(npz_folder, npz_file)
        video_path = os.path.join(video_folder, video_file)
        
        # Load video and get the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        cap.release()
        
        # Load skeleton data
        skeleton = load_skeleton_data(npz_path)
        
        # Plot video frame
        axs[0, i].imshow(frame)
        axs[0, i].axis('off')
        
        # Plot skeleton on frame
        plot_skeleton_on_image(axs[1, i], frame, skeleton)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
