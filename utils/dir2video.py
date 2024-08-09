import os  

# 设置图片目录和输出视频文件名  
image_dir = '/home/zlt/Documents/SkydivingPose/output/wholebody/54-p1-14'  
output_video = '/home/zlt/Documents/SkydivingPose/output/wholebody/54-p1-14.mp4'  

# 使用ffmpeg命令将图片合成视频  
# -start_number 设置从哪个编号开始  
# -i {image_dir}/%d.png 这里使用%d来匹配数字序列文件名  
# -r 30 设置帧率为30，如有需要可以调整  
# -pix_fmt yuv420p 是为了保证兼容性，一般用于处理H.264编码的mp4视频  
ffmpeg_cmd = f"ffmpeg -framerate 30 -start_number 0 -i {image_dir}/frame_%4d.png -pix_fmt yuv420p {output_video}"  

# 执行命令  
os.system(ffmpeg_cmd)  

print("Video generated successfully.")