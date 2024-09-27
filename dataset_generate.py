import os
import random
from PIL import Image, ImageChops, ImageEnhance
import math
import numpy as np
from PIL import ImageFilter
import cv2
import shutil


# 配置路径
root_dir = '/home/dfrobot/yolo/traffic/all_in_one/image/front_img'
background_folder = '/home/dfrobot/yolo/traffic/all_in_one/image/background'
# 数据集总图像数=每张背景图生成的组合图像数量*背景图数量
# 每张前景图图片随机生成的数量
random_image_count = 20
# 每张背景图生成的组合图像数量
image_output_count = 2 


# 获取根目录下的所有文件夹
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# 创建对应的 cropped_image_folders 和 output_folders
cropped_image_folders = {}
output_folders = {}

for folder in folders:
    cropped_image_folder = os.path.join('cropped_image', folder)
    output_folder = os.path.join('output', folder)
    
    # 创建文件夹
    os.makedirs(cropped_image_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    cropped_image_folders[folder] = cropped_image_folder
    output_folders[folder] = output_folder

# 剪裁透明部分
def trim(im):
    bg = Image.new(im.mode, im.size, (0, 0, 0, 0))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im

# 将图像变为320*320大小
def resize_image(image):
    return image.resize((320, 320))

# 向图像添加随机噪声
def add_noise(image):
    np_image = np.array(image)
    noise = np.random.randint(0, 1, (np_image.shape[0], np_image.shape[1], 4), dtype='uint8')
    np_image = np_image + noise
    return Image.fromarray(np.clip(np_image, 0, 255).astype('uint8'))

# 高斯模糊
def add_gaussian_blur(image):
    blur_radius = random.uniform(0, 1.5)
    return image.filter(ImageFilter.GaussianBlur(blur_radius))

# 添加运动模糊
def add_motion_blur(image):
    kernel_size = random.randint(5, 20)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    np_image = np.array(image)
    for i in range(3):
        np_image[:, :, i] = cv2.filter2D(np_image[:, :, i], -1, kernel)
    return Image.fromarray(np_image)

# 随机缩放图像
def random_scale(image):
    scale_factor = random.uniform(0.2, 1.4)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    return image.resize(new_size, Image.Resampling.LANCZOS)

# 随机旋转图像
def random_rotate(image):
    angle = random.uniform(-15, 15)
    return image.rotate(angle, expand=True)

# 随机调整亮度
def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.6, 1.3)
    return enhancer.enhance(factor)

# 随机改变图像的长宽比
def random_aspect_ratio(image):
    width_factor = random.uniform(0.8, 1)
    height_factor = random.uniform(0.9, 1.2)
    new_width = int(image.width * width_factor)
    new_height = int(image.height * height_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# 剪裁所有动物图像并保存
for image in folders:
    folder = os.path.join(root_dir, image)
    image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
    for image_file in image_files:
        img = Image.open(image_file).convert('RGBA')
        for i in range(random_image_count):
            processed_img = resize_image(img)
            processed_img = add_noise(processed_img)
            processed_img = random_scale(processed_img)
            processed_img = random_rotate(processed_img)
            processed_img = random_brightness(processed_img)
            processed_img = add_motion_blur(processed_img)
            processed_img = random_aspect_ratio(processed_img)
            processed_img = trim(processed_img)
            base_name = os.path.basename(image_file)
            name, ext = os.path.splitext(base_name)
            new_name = f"{name}_{i}{ext}"
            processed_img.save(os.path.join(cropped_image_folders[image], new_name))

print('剪裁完成')
print('--------------------------------')

print('开始生成合成图像')
print('--------------------------------')




# YOLO标注格式：class x_center y_center width height
def create_yolo_annotation(image_size, bbox, class_id):
    dw = 1. / image_size[0]
    dh = 1. / image_size[1]
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x_center *= dw
    y_center *= dh
    width *= dw
    height *= dh
    return f"{class_id} {x_center} {y_center} {width} {height}"

# 计算两个中心点之间的距离
def center_distance(bbox1, bbox2):
    x1_center = (bbox1[0] + bbox1[2]) / 2.0
    y1_center = (bbox1[1] + bbox1[3]) / 2.0
    x2_center = (bbox2[0] + bbox2[2]) / 2.0
    y2_center = (bbox2[1] + bbox2[3]) / 2.0
    return math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)

# 生成合成图像和标注
def generate_composite_image_and_annotation(background_img, image_imgs, image_classes, output_path, annotation_path):
    bg_width, bg_height = background_img.size
    annotations = []
    placed_bboxes = []

    for image_img, class_id in zip(image_imgs, image_classes):
        image_width, image_height = image_img.size
        max_x = bg_width - image_width
        max_y = bg_height - image_height
        if max_x < 0 or max_y < 0:
            print("Error: Background image is smaller than the front image.")
            continue

        for _ in range(100):  # 尝试最多100次以找到合适的位置
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            bbox = (x, y, x + image_width, y + image_height)
            if all(center_distance(bbox, placed_bbox) > 150 for placed_bbox in placed_bboxes):
                break
        else:
            # print("Warning: Could not find suitable position for front image.")
            continue

        # Paste the image image onto the background
        background_img.paste(image_img, (x, y), image_img)

        # 记录已放置的bbox
        placed_bboxes.append(bbox)

        # 转换为YOLO标注格式
        annotation = create_yolo_annotation((bg_width, bg_height), bbox, class_id)
        annotations.append(annotation)

    # Convert to RGB mode before saving
    background_img = background_img.convert('RGB')
    background_img.save(output_path, 'JPEG')

    # Save the annotations
    with open(annotation_path, 'w') as f:
        for ann in annotations:
            f.write(ann + '\n')

# 读取背景图和剪裁后的动物图像
# 注意这里要改图片后缀
backgrounds = [os.path.join(background_folder, f) for f in os.listdir(background_folder) if f.endswith('.jpg')]
print(backgrounds)
cropped_images = {image: [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')] for image, folder in cropped_image_folders.items()}

# 生成合成图像和标注
for image, image_files in cropped_images.items():
    for bg_file in backgrounds:
        background_img = Image.open(bg_file).convert('RGBA')
        
        for i in range(1, image_output_count):  # 每个背景图生成不同组合的图像数量
            num_images = random.randint(1, 2)
            selected_images = []
            selected_classes = []
            for _ in range(num_images):
                image_file = random.choice(image_files)
                selected_images.append(Image.open(image_file).convert('RGBA'))
                selected_classes.append(list(cropped_images.keys()).index(image))
            
            output_image_path = os.path.join(output_folders[image], f'composite_{os.path.splitext(os.path.basename(bg_file))[0]}_{i}.jpg')
            output_annotation_path = os.path.join(output_folders[image], f'composite_{os.path.splitext(os.path.basename(bg_file))[0]}_{i}.txt')
            generate_composite_image_and_annotation(background_img.copy(), selected_images, selected_classes, output_image_path, output_annotation_path)


# 创建muti文件夹路径
muti_output_folder = 'output/muti_output'
if not os.path.exists(muti_output_folder):
    os.makedirs(muti_output_folder)

# 生成muti文件夹中的图像和标注
for bg_file in backgrounds:
    background_img = Image.open(bg_file).convert('RGBA')
    # 随机选择三种不同的水果
    selected_image_types = random.sample(list(cropped_images.keys()), 3)
    selected_images = []
    selected_classes = []
    for image_type in selected_image_types:
        image_file = random.choice(cropped_images[image_type])
        selected_images.append(Image.open(image_file).convert('RGBA'))
        selected_classes.append(list(cropped_images.keys()).index(image_type))
    
    # 生成输出路径
    output_image_path = os.path.join(muti_output_folder, f'muti_{os.path.splitext(os.path.basename(bg_file))[0]}.jpg')
    output_annotation_path = os.path.join(muti_output_folder, f'muti_{os.path.splitext(os.path.basename(bg_file))[0]}.txt')
    generate_composite_image_and_annotation(background_img, selected_images, selected_classes, output_image_path, output_annotation_path)

print("muti文件夹中的图像和标注生成完毕！")

print("生成数据图像完毕！")
print('--------------------------------')

print('开始生成训练集和验证集')
print('--------------------------------')

# 创建路径
curr_path=os.getcwd()
imgtrainpath = os.path.join(curr_path,'images','train')
imgvalpath=os.path.join(curr_path,'images','validation')
imgtestpath=os.path.join(curr_path,'images','test')

labeltrainpath=os.path.join(curr_path,'labels','train')
labelvalpath=os.path.join(curr_path,'labels','validation')
labeltestpath=os.path.join(curr_path,'labels','test')


# 定义一个函数来清空或创建目录
def clear_or_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # 删除目录及其所有内容
    os.makedirs(path)  # 创建新的空目录

# 应用于所有路径
clear_or_create_dir(imgtrainpath)
clear_or_create_dir(imgvalpath)
clear_or_create_dir(imgtestpath)
clear_or_create_dir(labeltrainpath)
clear_or_create_dir(labelvalpath)
clear_or_create_dir(labeltestpath)

ip_datapath='output'

for dirname in os.listdir(ip_datapath):
    dirpath=os.path.join(ip_datapath, dirname)
    for file in os.listdir(dirpath):
        filepath=os.path.join(dirpath, file)
        newname=dirname+'_'+file
        if file.endswith((".txt")): # if label file, take it to label train path
            shutil.copy(filepath, labeltrainpath)
            path=os.path.join(labeltrainpath, file)
            newpath=os.path.join(labeltrainpath, newname)
        elif file.endswith((".jpg", ".JPG")): # if image file, resize and take it to image train path
            # img_resized=cv2.resize(cv2.imread(filepath), (image_size, image_size))
            img_resized=cv2.imread(filepath)  #这里删除了resize
            path=os.path.join(imgtrainpath, file)
            cv2.imwrite(path, img_resized)
            newpath=os.path.join(imgtrainpath, newname)
        os.rename(path, newpath) # Rename the file (label or image)

# moving 20% of data to validation

factor=0.2 

for file in random.sample(os.listdir(imgtrainpath), int(len(os.listdir(imgtrainpath))*factor)):
    basename=os.path.splitext(file)[0]
    textfilename=basename+'.txt'
    labelfilepath=os.path.join(labeltrainpath, textfilename)
    labeldestpath=os.path.join(labelvalpath, textfilename)
    imgfilepath=os.path.join(imgtrainpath, file)
    imgdestpath=os.path.join(imgvalpath, file)
    shutil.move(imgfilepath, imgdestpath)
    shutil.move(labelfilepath, labeldestpath)
    
newline='\n'

# Starting with a comment in config file
ln_1='# Train/val/test sets'+newline

# train, val and test path declaration
ln_2='train: ' +"'"+imgtrainpath+"'"+newline
ln_3='val: ' +"'" + imgvalpath+"'"+newline
ln_4='test: ' +"'" + imgtestpath+"'"+newline
ln_5=newline
ln_6='# Classes'+newline

# names of the classes declaration
ln_7='names:'+newline


# 自动生成分类
# class_lines = [f'  {i}: {folder}' + newline for i, folder in enumerate(folders)]
output_folder = 'output'  # 确保定义了output_folder

# 自动生成分类
class_lines = [f'  {i}: {os.path.join(output_folder, folder)}' + newline for i, folder in enumerate(folders)]


config_lines = [ln_1, ln_2, ln_3, ln_4, ln_5, ln_6, ln_7] + class_lines

config_path = os.path.join(curr_path, 'config.yaml')

with open(config_path, 'w') as f:
    f.writelines(config_lines)