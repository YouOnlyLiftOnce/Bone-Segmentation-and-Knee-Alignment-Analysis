import matplotlib.pyplot as plt
import pandas as pd

'''
plot the training/validation loss
'''

train_loss = pd.read_csv("train_loss.csv")
validation_loss = pd.read_csv("validation_loss.csv")

miou = pd.read_csv("mlou.csv")
dice_coefficient = pd.read_csv("dice_coefficient.csv")

# plot loss
fig = plt.figure(figsize=(15, 8))
x = train_loss["Step"]
ax1 = fig.add_subplot(111)
ax1.plot(x, train_loss["Value"], color="lightcoral", marker="*", label="training loss", alpha=0.8)
ax1.plot(x, validation_loss["Value"], color="cornflowerblue", marker="o", label="validation loss", alpha=0.8)
ax1.grid(linestyle="--", alpha=0.5)

ax1.set_xlabel("epochs", fontsize=20)
ax1.set_ylabel("loss", fontsize=20)
ax1.legend(fontsize=15)
ax1.set_title("Training Loss vs. Validation Loss", fontsize=20)
plt.show()

# plot loss
fig = plt.figure(figsize=(15, 8))
# fig.subplots_adjust(wspace=0.3, hspace=0.3)
x = train_loss["Step"]
ax1 = fig.add_subplot(111)
ax1.plot(x, miou["Value"], color="lightcoral", label="mIoU", alpha=0.8)
ax1.grid(linestyle="--", alpha=0.5)

ax1.set_xlabel("epochs", fontsize=20)
ax1.set_ylabel("mIoU", fontsize=20)
ax1.set_title("Validation mIoU", fontsize=20)
plt.show()

# plot loss
fig = plt.figure(figsize=(15, 8))
x = train_loss["Step"]
ax1 = fig.add_subplot(111)
ax1.plot(x, dice_coefficient["Value"], color="lightcoral", label="dice coefficient", alpha=0.8)
ax1.grid(linestyle="--", alpha=0.5)

ax1.set_xlabel("epochs", fontsize=20)
ax1.set_ylabel("dice coefficient", fontsize=20)
ax1.set_title("Validation Dice Coefficient", fontsize=20)
plt.show()
