"""
    main file

    based on: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
"""

import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import random_split

from tqdm import tqdm
from model import AudioClassifier_CNN
from dataset import Train_SoundDS, Eval_SoundDS

import matplotlib.pyplot as plt


# ----------------------------
# Fix random seed
# ----------------------------
RAND_SEED = 0
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
torch.cuda.manual_seed(RAND_SEED)
torch.backends.cudnn.deterministic = True


# ----------------------------
# Prepare data
# ----------------------------
def get_data_df(data_dir):
    data_n_class = []

    for file in glob.glob(os.path.join(data_dir, "*.wav")):
        if 'normal' in file:
            data_n_class.append([file, 1])
        elif 'void' in file:
            data_n_class.append([file, 0])
        else:  
            print("cannot identify file", file)

    df = pd.DataFrame(data_n_class, columns=['relative_path', 'classID'])

    return df


train_df = get_data_df('../train_set')
val_df = get_data_df('../val_set')
test_df = get_data_df('../test_set')



# ----------------------------
# Build training/validation/test sets
# ----------------------------
train_ds = Train_SoundDS(train_df)
val_ds = Eval_SoundDS(val_df)
test_ds = Eval_SoundDS(test_df)


# Create training, validation and test dataloaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
print("length of training, validation and test sets:", len(train_ds), len(val_ds), len(test_ds))


# ----------------------------
# Create the model
# ----------------------------
myModel = AudioClassifier_CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)


# ----------------------------
# Inference
# ----------------------------
def inference(model, val_dl, test=False):
    correct_prediction = 0
    total_prediction = 0

    label_list = []
    score_list = []

    # Disable gradient updates
    with torch.no_grad():
        for data in tqdm(val_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            score = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(score, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            label_list += labels.cpu().tolist()
            score_list += torch.sigmoid(score[:, 0]).cpu().tolist()

    acc = correct_prediction/total_prediction

    if test:
        return f'Accuracy: {acc:.4f}, Total items: {total_prediction}', score_list, label_list
    else:
        return f'Accuracy: {acc:.4f}, Total items: {total_prediction}'


# ----------------------------
# Training function
# ----------------------------
def training(model, train_dl, val_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(tqdm(train_dl)):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class 
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

        print("On val set:", inference(model, val_dl))

    print('Finished Training')


# Luanch here
training(myModel, train_dl, val_dl, num_epochs=60)

# test
test_result, score_list, label_list = inference(myModel, test_dl, test=True)
print("Test results:", test_result)


fpr, tpr, _ = roc_curve(label_list, score_list, pos_label=0)
auroc = auc(fpr, tpr)

print("auroc:", auroc)

plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)"%auroc)
plt.title('ROC Curve of CNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('CNN_auc.pdf', bbox_inches='tight')
plt.show()


