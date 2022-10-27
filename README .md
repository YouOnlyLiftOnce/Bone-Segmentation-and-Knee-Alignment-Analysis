# README

# Overview

It is a COMP8604 project repository. The goal of the project is to develop a pipeline for automated analysis of alignment in long-leg x-rays. There are two parts to this repository. First, a modified U- Net for segmenting the femur and tibia. Second, morphological methods for landmark locating.

# Quick start

- Install environments used in this project

```python
pip install requirements.txt
```

- Download the pre-trained model: https://anu365-my.sharepoint.com/:u:/g/personal/u7150030_anu_edu_au/ETkg7XHvHWdAnM1zNRCCYD0BFR2rWKT3oeMtJ57MD9qEQg?e=MrEgmY

Main files:

- [main.py](http://main.py) is the dashboard for training and testing the network. Supported by
- [inference.py](http://inference.py) is used for predicting the segmentation mask from a given x-ray
- [landmarks.py](http://landmarks.py) is used to construct mechanical alignment from a given x-ray
- hka_validation.py is used for overall alignment assessment.

# Data

- OAI 12-month visit data for training the network
- OAI 36-month visit data for HKA validation
- Only sample data are included in the repository. The complete dataset can be found:
    - OAI 12-month visit data with annotation: ‣
    - OAI 36-month visit data : ‣
- All data is in ./data directory.
    - ./data/raw/ stores the raw x-rays (bilateral)
    - ./data/preprocessed  stores all preprocessed images, corresponding masks(labels), and augmented data
    - ./data/train and ./data/test store the data for training and test, manually copy from the ./data/preprocessed
    - ./data/inference stores the data for inference, and the results of inference
    - ./data/HKA_validation stores the data and ground truth HKA values for validating the quality of the automated alignment

# Network Training

1. Download data from OAI 12-month visit data with annotation: ‣
2. Put raw data (x-rays) in  ./data/raw/  fold
3. Run [utils.preprocessing.py](http://utils.preprocessing.py) to preprocess the raw x-rays. The results are saved in ./data/preprocessed/images
4. Open Labelme to label the x-rays in ./data/preprocessed/images fold
5. Run [utils.json2mask.py](http://utils.json2mask.py) to convert the output of the labelme to masks. The masks are stored in ./data/preprocessed/masks
6. Manually divide and copy the x-rays and corresponding masks to the ./data/train and ./data/test.
7. Go to [main.py](http://main.py) set the hyperparameters for training, then run the main.py to train the network.
8. The training/validation loss is recorded with the tensorboard. Open it in the terminal to check the training process.
9. In [main.py](http://main.py) switch the mode to test the performance of the model on the test set

## Results of Training

|  | dice coefficient | mIoU | epoch |
| --- | --- | --- | --- |
| dc_loss | 0.978 | 0.982 | 52 |
| dc_loss(ig=2) | 0.972 | 0.979 | 47 |
| ce_loss | 0.969 | 0.976 | 67 |
| f_loss | 0.936 | 0.952 | 97 |
| ce+dc_loss | 0.969 | 0.976 | 24 |
| ce+dc_loss(ig=2) | 0.967 | 0.975 | 45 |
| f+dc_loss | 0.964 | 0.973 | 47 |
| f+dc_loss | 0.963 | 0.972 | 39 |
| f+ce_loss | 0.962 | 0.973 | 85 |
| f+ce+dc_loss | 0.968 | 0.975 | 47 |
| f+ce+dc_loss(ig=2) | 0.963 | 0.972 | 29 |
- dc_loss = dice loss
- dc_loss(ig=2) = dice loss with ignoring background pixels
- ce_loss = cross-entropy loss
- f_loss = focal loss

 Results on test set: dice coefficient: 0.972, mIoU: 0.980

![loss.png](README/loss.png)

![dice_coefficient.png](README/dice_coefficient.png)

![miou.png](README/miou.png)

# Automated Mechanical Alignment with Assessment

1. Download data from OAI 36-month visit data: ‣
    1. the .zip file includes the x-rays and the ground truth HKA
2. Put data in ./data/HKA_validation
3. Run the hka_validation.py
    1. the data of ground truth HKA and predicted HKA is stored in true_hka.npy and pred_hka.npy.
    2. delete those two files for re-assessment.
    3. the top k best/worst alignment is saved in ./data/HKA_validation/best_alignment and data/HKA_validation/worst_alignment for the analysis purpose.

## Quantitive Results of Assessment

difference mean: 0.5835071292298868, 

difference variance: 0.26797344352371594, 

MSE: 0.60845401338582, 

paired t-test: Ttest_relResult(statistic=-0.6939754226448148, pvalue=0.48817010111589665)

![HKA_measurement.png](README/HKA_measurement.png)

## Qualitative Results of Assessment

![HKA results.png](README/HKA_results.png)