from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import os

if __name__ == "__main__":

    folder_ori = "./output/testset/truth"
    folder_path1 = "/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/output/0.original_demo"
    folder_path2 = "/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/output/result_9"

    # image_files_ori = [f for f in os.listdir(folder_ori) if f.endswith(('.png', '.jpg'))]
    image_files = [f for f in os.listdir(folder_path1) if f.endswith(('.png', '.jpg'))]

    j = 0
    total = 0

    ## truth v.s original model
    # for i in range(1 , len(image_files) , 2):
    #     img_ori_path = folder_ori + "/" + image_files_ori[j]
    #     img1_path = folder_path1 + "/" + image_files[i]

    #     img_ori = np.array(Image.open(img_ori_path).convert("1"))
    #     img1 = np.array(Image.open(img1_path).convert("1"))

    #     total += psnr(img_ori , img1)
    #     print(psnr(img_ori , img1))

    #     j += 1

    ## original model v.s model_TSM
    for i in range(5):
        img1_path = folder_path1 + "/" + str(i+1) + '_pred.jpg'
        img2_path = folder_path2 + "/" + str(i+1) + '_pred.jpg'
        # print(img1_path, img2_path)

        img1 = np.array(Image.open(img1_path).convert("1"))
        img2 = np.array(Image.open(img2_path).convert("1"))

        total += psnr(img1 , img2)
        print(psnr(img1 , img2))
    # img1_path = "/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/output/0.original_demo/1_pred.jpg"
    # img2_path = '/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/output/result_9/1_pred.jpg'

    # img1 = np.array(Image.open(img1_path).convert("1"))
    # img2 = np.array(Image.open(img2_path).convert("1"))

    # total += psnr(img1 , img2)
    # print(psnr(img1 , img2))

    print(f"Average : {total / 5}")