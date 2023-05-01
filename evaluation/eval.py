import cv2

def compare_ssim(original_img, swapped_img):
    # Convert the images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    swapped_gray = cv2.cvtColor(swapped_img, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM between the original and swapped images
    ssim_score = compare_ssim(original_gray, swapped_gray)
    return ssim_score
