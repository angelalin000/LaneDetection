from PIL import Image
import numpy as np

# Pixel-by-pixel Comparison
def calculate_similarity(image_path1, image_path2):
    # open two images
    img1 = Image.open(image_path1).convert('L')  # graysacle
    img2 = Image.open(image_path2).convert('L')  # graysacle

    # transform to np
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # ensure the size of two images are the same
    if img1_array.shape != img2_array.shape:
        raise ValueError("The images must have the same dimensions.")

    # cal the same pixel
    identical_pixels = np.sum(img1_array == img2_array)
    total_pixels = img1_array.size

    # cal accuracy
    similarity = (identical_pixels / total_pixels) * 100

    return similarity


# file path
# for i in range(5):
image1 = '/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/output/result_1/1_pred.jpg'
image2 = '/content/drive/MyDrive/graduation_project4/Robust-Lane-Detection/LaneDetectionCode/data/testset/truth/1_13.jpg'
similarity = calculate_similarity(image2, image1)
print(f"Similarity: {similarity:.2f}%")
