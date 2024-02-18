# Extended Maximum intensity Projection (EMIP)

This code is used to convert 3D multi-channel images into 2D images using Extended Maximum intensity Projection (EMIP) for training LECL model. 

An example of 2D image generated using EMIP from the multi-channel fluorescent images

<p align="center">
<img src="img.png" alt="alt text" width="400" />
</p>

An example of generated cluster mask using point annotations.

<p align="center">
<img src="mask.png" alt="alt text" width="400" />
</p>

1. purple => Background
2. yellow => Nuclei with the marker
3. orange => Nuclei without the marker
4. black => Unlabeled area

## Usage

To use the code on a new dataset, please follow the stages below:

1- Please save the data, including the fluorescent images in TIFF format and the associated Region of Interest (ROI) files in ROI format, within directories named "TIFF_wsi" and "LesionROI". 
Each pair of fluorescent and ROI images should have identical names and should not contain underscores (i.e., '_').
Please ensure that your directory tree exactly matches the following chart:

```none
├── dataset
├── LesionROI
├── TIFF_wsi
```

2-Please run the command below

python main.py 

3- The prepared data for training will be saved in the directories below:

```none
├── dataset
│   ├── data_for_train
│   │   ├── LECL_data
│   │   │   ├── train
│   │   │   ├── test
│   │   │   ├── valid
```

# Citation

Moradinasab, N., Deaton, R.A., Shankman, L.S., Owens, G.K. and Brown, D.E., 2023, October. Label-efficient Contrastive Learning-based model for nuclei detection and classification in 3D Cardiovascular Immunofluorescent Images. In Workshop on Medical Image Learning with Limited and Noisy Data (pp. 24-34). Cham: Springer Nature Switzerland.