### SSIM (Structural Similarity Index) 
SSIM is a widely used image quality metric that measures the similarity between two images based on their structural information. The metric was designed to model the human perception of image quality, and it is particularly useful for evaluating the performance of image processing algorithms such as image compression or image enhancement techniques.

The SSIM index ranges from -1 to 1, where a score of 1 indicates that the two images are identical, and a score of -1 indicates that they are completely dissimilar. A score of 0 indicates that the two images are uncorrelated.

The SSIM metric consists of three components: luminance, contrast, and structure. The luminance component measures the overall brightness of the image, while the contrast component measures the difference in brightness between different parts of the image. The structure component measures the correlation between the luminance and contrast of different parts of the image.

To calculate the SSIM index between two images, the images are first divided into small windows, and the SSIM score is calculated for each window. The window scores are then combined to produce a single SSIM score for the entire image.

SSIM is a widely used metric in computer vision and image processing applications because it provides a more accurate measure of image quality than other metrics such as mean squared error (MSE) or peak signal-to-noise ratio (PSNR), which only measure the difference in pixel values between two images.

### PSNR (Peak Signal-to-Noise Ratio)
PSNR is a measure of the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the quality of the signal. In image processing, it is used to measure the quality of the reconstructed image after compression or other processing. Higher PSNR values indicate higher image quality.

### MSE (Mean Squared Error)
MSE measures the average squared differences between two images. It's calculated by comparing each pixel value in the original and processed image and taking the square of the difference between them. A smaller MSE value indicates better image quality.