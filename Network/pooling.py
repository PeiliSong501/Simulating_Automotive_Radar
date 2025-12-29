import os

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from DisNet import DisNet, KLDivergenceLoss
# 输入和输出目录
input_dir = r"D:\VoD_dataset\view_of_delft_PUBLIC\radar\training\PMapDataset\pmap_image"
output_dir = r"D:\VoD_dataset\view_of_delft_PUBLIC\radar\training\PMapDataset\pmap_image_pooling"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# #遍历输入目录中的所有图像
# for img_name in os.listdir(input_dir):
#     if img_name.endswith(".jpg"):
#         # 加载图像并转换为灰度图
#         img_path = os.path.join(input_dir, img_name)
#         print(img_path)
#         img = Image.open(img_path).convert('L')  # 转为灰度图
#         img_np = np.array(img)
#
#         # 转换为张量并添加批次和通道维度
#         img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#
#         # 使用自定义块池化操作
#         kernel_size = (40, 40)  # 定义块大小
#         stride = (40, 40)
#         pooled_tensor = F.max_pool2d(img_tensor, kernel_size=kernel_size, stride=stride, ceil_mode=True)
#
#         # 恢复原始尺寸
#         # pooled_tensor_resized = F.interpolate(pooled_tensor, size=img_np.shape, mode='nearest')
#
#         # 将结果保存为图像
#         # pooled_img_np = pooled_tensor_resized.squeeze().numpy().astype(np.uint8)
#         pooled_img_np = pooled_tensor.squeeze().numpy().astype(np.uint8)
#         pooled_img = Image.fromarray(pooled_img_np)
#         print(pooled_img_np.shape)
#         pooled_img.save(os.path.join(output_dir, img_name))
#
# print("所有图像处理完成并保存到目标目录。")

# img1 = np.array(cv2.imread(os.path.join(input_dir,"0.jpg")))
# img2 = np.array(cv2.imread(os.path.join(output_dir,"0.jpg")))
# img1 = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
# img2 = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).permute(0,3,1,2)
# print(img1.shape, img2.shape)
# pooled_tensor_resized = F.interpolate(img2, size=img1.shape[2:], mode='nearest')
# kl_loss = KLDivergenceLoss()
# print(kl_loss(img1,pooled_tensor_resized))

img1 = cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/pmap_image/7335.jpg',cv2.IMREAD_GRAYSCALE)
img1 = torch.tensor(img1).unsqueeze(0).unsqueeze(0)

img2 = np.array(cv2.imread('D:/VoD_dataset/view_of_delft_PUBLIC/radar/training/PMapDataset/test_image/7335.jpg', cv2.IMREAD_GRAYSCALE))
img2 = torch.tensor(img2).unsqueeze(0).unsqueeze(0)
non_black_pixels = np.count_nonzero(img2)
print(img2.shape)
# 计算总像素点数量
total_pixels = img2.size

# 计算非纯黑像素点的比例
# non_black_ratio = non_black_pixels / total_pixels
# print(non_black_pixels,total_pixels)
# print(np.mean(img2),np.max(img2), np.min(img2))
# print(f"非纯黑像素点比例: {non_black_ratio:.4f}")

pooled_tensor_resized = F.interpolate(img2, size=img1.shape[2:], mode='nearest')


kl_loss = KLDivergenceLoss()
print(kl_loss(img1,pooled_tensor_resized))
pooled_tensor_resized = pooled_tensor_resized.squeeze(0).squeeze(0).cpu().detach().numpy()
cv2.imshow('generated',pooled_tensor_resized)
cv2.waitKey(0)