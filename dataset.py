from torch.utils.data import Dataset
from PIL import Image
import torch
import config
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
import cv2
import os

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = self.transforms(data)
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        for i in range(5):
            data.append(torch.unsqueeze(self.transforms(Image.open(img_path_list[i])), dim=0))
        data = torch.cat(data, 0)
        label = Image.open(img_path_list[5])
        label = torch.squeeze(self.transforms(label))
        sample = {'data': data, 'label': label}
        return sample

# def extract_frames(video_path, batch_idx):

#     # 讀取影片
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     #print(f"Total frames: {frame_count}")

#     for i in range(batch_idx):
#         ret, frame = cap.read()
#         if not ret:
#             break
#     # 讀取當前幀
#     ret, frame = cap.read()
#     #height, width = frame.shape[:2]
#     frame_1 = cv2.resize(frame, (256, 128))

#     cap.release()
#     return frame_1
#     #print("完成影片幀提取")

# def extract_frames(video_path, output_folder):
#     # 確保輸出資料夾存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 讀取影片
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print(f"Total frames: {frame_count}")

#     frame_idx = 0
#     frame_queue = queue.Queue(maxsize=6)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         height, width = frame.shape[:2]
#         frame = cv2.resize(frame, (256, 128))
#         frame = cv2.resize(frame, (width, height))
#         # 定義圖片檔名
#         frame_filename = os.path.join(output_folder, f"frame_{frame_idx:04d}.jpg")
#         # 保存圖片
#         cv2.imwrite(frame_filename, frame)
#         frame_queue.put(frame)
#         if frame_queue.full():
#             frames = [frame_queue.get() for _ in range(5)]
#             for i in range(5):
#                 frame = frames[i]
#                 data.append(torch.unsqueeze(self.transforms(frame), dim=0))
#             # print(result)
#             # 將第一幀重新放回佇列，這樣可以滑動窗口
#             d
#             frame_queue.put(frames[-1])
        
#         frame_idx += 1

#     cap.release()
#     print("完成影片幀提取")

# class RoadSequenceDatasetList(Dataset):

#     def __init__(self, file_path, transforms):
#         self.file_path = file_path
#         #self.img_list = readTxt(file_path)
#         cap = cv2.VideoCapture(file_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.dataset_size = total_frames
#         self.transforms = transforms
#     def __len__(self):
#         return self.dataset_size

#     def __getitem__(self, idx):
#         #print(f'idx:{idx}')
#         idx -= 1
#         #img_path_list = self.img_list[idx]
#         file_path = self.file_path
#         data = []
#         for i in range(5):
#             frame = extract_frames(file_path, idx)
#             data.append(torch.unsqueeze(self.transforms(frame), dim=0))
#         data = torch.cat(data, 0)
#         sample = data
#         return sample


