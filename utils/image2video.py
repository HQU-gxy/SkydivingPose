import cv2 
from pathlib import Path

IMG_FOLDER=Path("/home/zlt/Documents/SkydivingPose/output/aaa/pose2D")

imgs = IMG_FOLDER.glob("*.png")

writer = cv2.VideoWriter("aaa-o.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, (1080, 1920))

for img in imgs:
    if img.name == "0073_2D.png":
        break
    i = cv2.imread(str(img))
    writer.write(i)

writer.release()
