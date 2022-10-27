import pandas as pd
from landmarks import locate_mechanical_axis, draw_mechanical_axis
import torch
from inference import inference
import os
import cv2 as cv
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import shutil

'''
hka validation.
'''
# link filename to ground truth hka
hka_df = pd.read_excel("./data/HKA_validation/36m_hka.xlsx")
patient_df = pd.read_excel("./data/HKA_validation/patient_id.xlsx")
hka_df = hka_df.rename(columns={"V05HKANGLE": "HKA"})
patient_df = patient_df.rename(columns={"Corresponding patient ID": "ID"})
join_df = pd.merge(hka_df, patient_df, on='ID', how='inner')
file_hka_df = join_df[["ID", 'Filename', "SIDE", "HKA"]].dropna()
file_hka_right = file_hka_df.loc[file_hka_df['SIDE'] == 1]
file_hka_left = file_hka_df.loc[file_hka_df['SIDE'] == 2]
file_hka_dict = file_hka_right[["Filename", "HKA"]].set_index("Filename").T.to_dict('records')[0]

img_path = "./data/HKA_validation/36m_images/"
mask_path = "./data/HKA_validation/36m_masks/"
save_path = "./data/HKA_validation/36m_alignment/"
model_path = "exp_1_128x1024_dc_b=8_model.pth"
img_names = os.listdir(img_path)

true_hka = []
pred_hka = []

if os.path.exists("true_hka.npy") and os.path.exists("pred_hka.npy"):

    pred_hka = np.load("pred_hka.npy")
    true_hka = np.load("true_hka.npy")

else:

    model = torch.load(model_path)
    inference(img_path, model, save_path=mask_path)
    img_names = os.listdir(img_path)

    for name in tqdm(img_names):
        img = cv.imread(img_path + name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(mask_path + name[:-4] + '.png')
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
        _, _, _, _, _, hka = locate_mechanical_axis(mask)
        print(name)
        draw_mechanical_axis(img, mask, img_name=name, save_dir=save_path)

        pred_hka.append(hka)
        true_hka.append(float(file_hka_dict[name]))

    assert len(pred_hka) == len(true_hka)

    pred_hka = np.array(pred_hka)
    true_hka = np.array(true_hka)
    np.save("./pred_hka.npy", pred_hka)
    np.save("./true_hka.npy", true_hka)

mead_diff = np.mean(np.abs(pred_hka - true_hka))
var_diff = np.var(np.abs(pred_hka - true_hka))
mse = mean_squared_error(pred_hka, true_hka)

# pair test
t_res = stats.ttest_rel(pred_hka, true_hka)

print(f"difference mean: {mead_diff}, difference variance: {var_diff}, MSE: {mse}, paired t-test: {t_res}")

# find k best and worst alignment results
best_dir = "./data/HKA_validation/best_alignment/"
worst_dir = "./data/HKA_validation/worst_alignment/"
k = 10
best_idx = np.argsort(np.abs(pred_hka - true_hka))[:k].tolist()
worst_idx = np.argsort(np.abs(pred_hka - true_hka))[-k:].tolist()
name_array = np.array(img_names)
best_img = name_array[best_idx]
worst_img = name_array[worst_idx]

for i in range(len(best_img)):
    shutil.copyfile(save_path + best_img[i], best_dir + best_img[i])
    shutil.copyfile(save_path + worst_img[i], worst_dir + worst_img[i])

# check Normality
fig = plt.figure(figsize=(15, 10))
# fig.subplots_adjust(wspace=0.3, hspace=0.3)
y = range(len(pred_hka))
ax1 = fig.add_subplot(111)
ax1.scatter(y, pred_hka, color="lightcoral", marker="^", label="prediction", alpha=0.8)
ax1.scatter(y, true_hka, color="cornflowerblue", marker="o", label="ground truth", alpha=0.8)
ax1.grid(linestyle="--", alpha=0.5)

ax1.set_xlabel("example index", fontsize=20)
ax1.set_ylabel("HKA", fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title("HKA Measurements: Prediction vs. Ground truth", fontsize=20)
plt.show()
