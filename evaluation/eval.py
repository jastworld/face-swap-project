import cv2
from skimage.metrics import structural_similarity, simple_metrics
import os
import numpy as np
from tabulate import tabulate


def compare_images(original_img, swapped_img):
    # Convert the images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    swapped_gray = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2GRAY)

    ssim = structural_similarity(original_gray, swapped_gray)
    psnr = simple_metrics.peak_signal_noise_ratio(original_gray, swapped_gray)
    mse = simple_metrics.mean_squared_error(original_gray, swapped_gray)
    return ssim, psnr, mse

def get_images_path(dir_path):
    paths = []
    for filename in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, filename)):
            paths.append(os.path.join(dir_path, filename))
    paths.sort()
    return paths

def reshape_images(images):
    min_shape = np.array([999999999, 999999999, 999999999])
    for image in images:
        mask = image.shape < min_shape
        min_shape = image.shape * mask + min_shape * (1-mask)

    new_shape = (min_shape[0], min_shape[1])
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], new_shape)
    return images

source_directory = ".\image\source"
direct_cut_directory = ".\image\\result\direct_cut"
poisson_blend_directory = ".\image\\result\poisson_blend"
source_images = get_images_path(source_directory)
direct_cut_images = get_images_path(direct_cut_directory)
poisson_blend_images = get_images_path(poisson_blend_directory)

N = len(source_images)
assert N == len(direct_cut_images) == len(poisson_blend_images)
evaluation = {
    "Images":[],
    "Direct Cut SSIM": [],
    "Direct Cut PSNR": [],
    "Direct Cut MSE": [],
    "Poisson Blend SSIM": [],
    "Poisson Blend PSNR": [],
    "Poisson Blend MSE": [],
}
for i in range(N):
    images = [cv2.imread(source_images[i]), cv2.imread(direct_cut_images[i]), cv2.imread(poisson_blend_images[i])]
    if images[0] is None or images[1] is None or images[2] is None:
        continue
    src_img, direct_cut_img, poisson_blend_img= reshape_images(images)
    
    assert src_img.shape == direct_cut_img.shape == poisson_blend_img.shape
    evaluation["Images"].append(source_images[i])
    ssim, psnr, mse = compare_images(src_img, direct_cut_img)
    evaluation["Direct Cut SSIM"].append(ssim)
    evaluation["Direct Cut PSNR"].append(psnr)
    evaluation["Direct Cut MSE"].append(mse)
    ssim, psnr, mse = compare_images(src_img, poisson_blend_img)
    evaluation["Poisson Blend SSIM"].append(ssim)
    evaluation["Poisson Blend PSNR"].append(psnr)
    evaluation["Poisson Blend MSE"].append(mse)

print(tabulate(evaluation, headers='keys'))

# # Load the source and target images
# src_img = cv2.imread("path/to/source/image.jpg")
# tgt_img = cv2.imread("path/to/target/image.jpg")
