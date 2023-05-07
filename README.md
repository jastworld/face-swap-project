# face-swap-project

**Usage**<br />
main.py [-h] --detector DETECTOR --image1 IMAGE1 --image2 IMAGE2 --output1 OUTPUT1 --output2 OUTPUT2 --swapper SWAPPER

- Detector: detector mode (0 - dlib detector / 1 - cascade detector)\
- Image1: Source Image file path\
- Image2: Target Image file path\
- Output1: Poisson Blend output file path\
- Output2: Simple merge output file path\
- Swapper: swapper mode (0-warps the source image to fit target face 1-warps the target image to fit source face)


**ex)** <br />
```$ python main.py --detector 0 --image1 ../image/source/img1.jpg --image2 ../image/target/img1.jpg --output1 ../image/result/poisson_blend_target_warp/img1.jpg --output2 ../image/result/direct_cut_target_warp/img1.jpg --swapper 0```

**File Structure** <br />
```bash
├── data (#pre-trained data files for detection)
├── evaluation (#evaluation code)
├── src (#main code to run face swap)
├── images (#input and output images)
│   ├── source
│   ├── target
│   ├── result
│   │   ├── direct_cut_sorce_warp
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ..
│   │   ├── direct_cut_target_warp
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ..
│   │   ├── poisson_blend_sorce_warp
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ..
│   │   ├── poisson_blend_target_warp
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ..
├── README.md
├── requirement.txt
├── .DS_Store
└── .gitignore
```


## Evaluation

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




### How to run the evaluation of the swapping techniques
1. Go to ./evaluation folder and modify the directory of files you want to evaluate
2. Ensure that the images are as follows source/1.jpg + target/1.jpg = result/1.jpg 
```
source_directory = ".\image\source"
target_directory = ".\image\\target"
direct_cut_directory = ".\image\\result\direct_cut_source_swap"
poisson_blend_directory = ".\image\\result\poisson_blend_source_swap"
```
`$ python evaluation/evaluation_script.py`