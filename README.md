# [ECCV2024] Just a Hint: Point-Supervised Camouflaged Object Detection


![Framework](figure/Framework.png)


# Prerequisites
- python 3.6
- Pytorch 1.12.1
- Torchvision 0.13.1
- Scikit_image=0.19.2
- Skimage=0.0

# Download P-COD Dataset

- Point supervised PCOD: [google](https://drive.google.com/file/d/17oa6-IU2Dr9Q1KKQ74UoL0hoFd5F7bOd/view?usp=sharing)

# Train
- Generate square clusters (d*d) by point label, [d*d cluster](https://drive.google.com/file/d/1L6l5ijona7J5eX5tX8aGSjwCY1oBdV7L/view?usp=drive_link) , After that training generates [Pred_Map](https://drive.google.com/file/d/1RjgNvc83wnTKAaVcRFg7gxVY85771XGg/view?usp=drive_link) ,then , run Point2gt.py to generate the [final supervisor labels](https://drive.google.com/file/d/1_la4aF9VMv_VG3pQIhc1PXNJa8dxIn26/view?usp=drive_link) .
- Download training dataset and testing dataset . Put them in the right path.
- Using the labels generated above for training ( "1" stands for foregrounds, "2" for backgrounds, and "0" for unlabeled regions. (The image is viewed as black because its range is 0-255)) .
- Run python train.py.

# Test and Evaluate 
- Modify path and filename.
- Run python test.py
  
# Experimental Results
![result](figure/Result.png)
