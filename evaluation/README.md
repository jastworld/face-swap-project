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

Source Swapp Image Results

| Images    |   Direct Cut SSIM |   Direct Cut PSNR |   Direct Cut MSE |   Poisson Blend SSIM |   Poisson Blend PSNR |   Poisson Blend MSE |
|:----------|------------------:|------------------:|-----------------:|---------------------:|---------------------:|--------------------:|
| img1.jpg  |          0.904094 |           24.159  |         249.56   |             0.910873 |              26.9375 |             131.622 |
| img2.jpg  |          0.928226 |           24.2052 |         246.923  |             0.944653 |              27.9288 |             104.762 |
| img3.jpg  |          0.902647 |           22.5423 |         362.115  |             0.903182 |              23.1357 |             315.874 |
| img4.jpg  |          0.922494 |           22.3367 |         379.675  |             0.915381 |              18.7713 |             862.887 |
| img5.jpg  |          0.926466 |           28.0502 |         101.872  |             0.925259 |              25.9772 |             164.194 |
| img6.jpg  |          0.921525 |           24.984  |         206.385  |             0.892842 |              19.6824 |             699.58  |
| img7.jpg  |          0.929192 |           23.0075 |         325.335  |             0.894777 |              19.035  |             812.044 |
| img8.jpg  |          0.938    |           27.9016 |         105.419  |             0.888542 |              19.3643 |             752.745 |
| img9.jpg  |          0.941471 |           26.7991 |         135.885  |             0.940349 |              24.4424 |             233.796 |
| img10.jpg |          0.87579  |           23.1712 |         313.302  |             0.852899 |              17.8595 |            1064.47  |
| img11.jpg |          0.947102 |           28.5278 |          91.2651 |             0.902782 |              17.1688 |            1247.97  |
| img12.jpg |          0.87196  |           22.903  |         333.258  |             0.84758  |              16.9716 |            1305.92  |
| img13.jpg |          0.869748 |           22.4758 |         367.706  |             0.878129 |              25.1339 |             199.382 |
| img14.jpg |          0.915595 |           24.6128 |         224.802  |             0.919393 |              25.8939 |             167.374 |
| img15.jpg |          0.8712   |           23.7632 |         273.374  |             0.858255 |              17.4012 |            1182.92  |
| img16.jpg |          0.870806 |           22.7655 |         343.979  |             0.874537 |              23.5916 |             284.394 |
| img17.jpg |          0.86877  |           24.3145 |         240.784  |             0.845795 |              15.686  |            1755.83  |
| img18.jpg |          0.868753 |           20.8263 |         537.593  |             0.864383 |              18.0596 |            1016.52  |
| Average   |          0.904102 |           24.297  |         268.846  |             0.892201 |              21.28   |             683.46  |

Target Swap Images

| Images    |   Direct Cut SSIM |   Direct Cut PSNR |   Direct Cut MSE |   Poisson Blend SSIM |   Poisson Blend PSNR |   Poisson Blend MSE |
|:----------|------------------:|------------------:|-----------------:|---------------------:|---------------------:|--------------------:|
| img1.jpg  |          0.742468 |           19.3457 |          755.978 |             0.744616 |              19.8052 |             680.085 |
| img2.jpg  |          0.795204 |           20.7903 |          542.067 |             0.800036 |              21.1549 |             498.417 |
| img3.jpg  |          0.677108 |           18.528  |          912.59  |             0.678382 |              18.6128 |             894.943 |
| img4.jpg  |          0.811359 |           19.4763 |          733.579 |             0.8027   |              20.899  |             528.667 |
| img5.jpg  |          0.68118  |           17.1519 |         1252.84  |             0.684977 |              17.4948 |            1157.71  |
| img6.jpg  |          0.801211 |           19.8956 |          666.064 |             0.783414 |              19.0137 |             816.033 |
| img7.jpg  |          0.818234 |           19.5302 |          724.534 |             0.816023 |              19.7555 |             687.911 |
| img8.jpg  |          0.804039 |           20.606  |          565.566 |             0.807582 |              20.8299 |             537.142 |
| img9.jpg  |          0.67203  |           20.5474 |          573.248 |             0.673998 |              20.4229 |             589.917 |
| img10.jpg |          0.728386 |           19.0127 |          816.23  |             0.728243 |              19.445  |             738.884 |
| img11.jpg |          0.636317 |           21.5567 |          454.366 |             0.634905 |              21.6619 |             443.501 |
| img13.jpg |          0.745005 |           18.6077 |          896.009 |             0.755785 |              19.2468 |             773.386 |
| img14.jpg |          0.778487 |           23.7072 |          276.923 |             0.78105  |              25.5802 |             179.913 |
| img15.jpg |          0.757574 |           19.3119 |          761.887 |             0.757905 |              19.658  |             703.521 |
| img16.jpg |          0.774291 |           19.4979 |          729.944 |             0.778078 |              19.7805 |             683.957 |
| img17.jpg |          0.723573 |           17.6979 |         1104.83  |             0.700135 |              17.8935 |            1056.16  |
| img18.jpg |          0.755603 |           17.6214 |         1124.45  |             0.762962 |              18.9528 |             827.568 |
| Average   |          0.747181 |           19.5815 |          758.3   |             0.746517 |              20.0122 |             693.983 |




### How to run the evaluation
1. Go to ./evaluation folder and modify the directory of files you want to evaluate
2. Ensure that the images are as follows source/1.jpg + target/1.jpg = result/1.jpg 
```
source_directory = ".\image\source"
target_directory = ".\image\\target"
direct_cut_directory = ".\image\\result\direct_cut_source_swap"
poisson_blend_directory = ".\image\\result\poisson_blend_source_swap"
```
`$ python evaluation/evaluation_script.py`