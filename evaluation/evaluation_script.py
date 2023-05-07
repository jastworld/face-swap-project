import math

import cv2
import os
import numpy as np
from tabulate import tabulate


def ssim_fn(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse)), mse

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_fn(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def compare_images(original_img, swapped_img):
    # Convert the images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    swapped_gray = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2GRAY)

    ssim = calculate_ssim(original_gray, swapped_gray)
    psnr, mse = calculate_psnr(original_gray, swapped_gray)
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
        min_shape = image.shape * mask + min_shape * (1 - mask)

    new_shape = (min_shape[0], min_shape[1])
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], new_shape)
    return images

if __name__ == "__main__":
    source_directory = ".\image\source"
    target_directory = ".\image\\target"
    direct_cut_directory = ".\image\\result\direct_cut_source_swap"
    poisson_blend_directory = ".\image\\result\poisson_blend_source_swap"
    source_images = get_images_path(source_directory)
    target_images = get_images_path(target_directory)

    direct_cut_images = get_images_path(direct_cut_directory)
    poisson_blend_images = get_images_path(poisson_blend_directory)
    print(direct_cut_images)

    N = len(target_images)
    print(N, len(direct_cut_images), len(poisson_blend_images))
    assert N == len(direct_cut_images) == len(poisson_blend_images)
    evaluation = {
        "Images": [],
        "Direct Cut SSIM": [],
        "Direct Cut PSNR": [],
        "Direct Cut MSE": [],
        "Poisson Blend SSIM": [],
        "Poisson Blend PSNR": [],
        "Poisson Blend MSE": [],
    }
    for i in range(N):
        if target_images[i].split("\\")[-1] == direct_cut_images[i].split("\\")[-1] ==  poisson_blend_images[i].split("\\")[-1]:
            pass
        else:
            continue
        images = [cv2.imread(target_images[i]), cv2.imread(direct_cut_images[i]), cv2.imread(poisson_blend_images[i])]
        if images[0] is None or images[1] is None or images[2] is None:
            continue
        src_img, direct_cut_img, poisson_blend_img = reshape_images(images)

        assert src_img.shape == direct_cut_img.shape == poisson_blend_img.shape
        ssim, psnr, mse = compare_images(src_img, direct_cut_img)
        if ssim == 1: # skip cos it adds noise
            continue
        evaluation["Images"].append(target_images[i].split("\\")[-1])

        evaluation["Direct Cut SSIM"].append(ssim)
        evaluation["Direct Cut PSNR"].append(psnr)
        evaluation["Direct Cut MSE"].append(mse)
        ssim, psnr, mse = compare_images(src_img, poisson_blend_img)
        evaluation["Poisson Blend SSIM"].append(ssim)
        evaluation["Poisson Blend PSNR"].append(psnr)
        evaluation["Poisson Blend MSE"].append(mse)

    evaluation["Images"].append("Average")

    evaluation["Direct Cut SSIM"].append(np.average(evaluation["Direct Cut SSIM"]))
    evaluation["Direct Cut PSNR"].append(np.average(evaluation["Direct Cut PSNR"]))
    evaluation["Direct Cut MSE"].append(np.average(evaluation["Direct Cut MSE"]))
    evaluation["Poisson Blend SSIM"].append(np.average(evaluation["Poisson Blend SSIM"]))
    evaluation["Poisson Blend PSNR"].append(np.average(evaluation["Poisson Blend PSNR"]))
    evaluation["Poisson Blend MSE"].append(np.average(evaluation["Poisson Blend MSE"]))
    print(tabulate(evaluation, headers='keys', tablefmt="pipe"))
