# face-swap-project

Usage: main.py [-h] --detector DETECTOR --image1 IMAGE1 --image2 IMAGE2 --output1 OUTPUT1 --output2 OUTPUT2 --swapper SWAPPER

Detector: detector mode (0 - dlib detector / 1 - cascade detector)
Image1: Source Image file path
Image2: Target Image file path
Output1: Poisson Blend output file path
Output2: Simple merge output file path
Swapper: swapper mode (0-warps the source image to fit target face 1-warps the target image to fit source face)


ex) $ python main.py --detector 0 --image1 ../image/source/img1.jpg --image2 ../image/target/img1.jpg --output1 ../image/result/poisson_blend_target_warp/img1.jpg --output2 ../image/result/direct_cut_target_warp/img1.jpg --swapper 0

File Structure

data/ - pre-trained data files for detection
evaluation/ - evaluation code
src/ - main code to run face swap
image/ - input and output images
source/
target/ 
result/
direct_cut_source_warp/
img1.jpg
img2.jpg
..
direct_cut_target_warp/
img1.jpg
img2.jpg
..
poisson_blend_source_warp/
img1.jpg
img2.jpg
..
poisson_blend_target_warp/
img1.jpg
img2.jpg
..
