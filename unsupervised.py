"""
    unsupervised method, here we use PCA
"""

import glob, os
from audio_utils import AudioUtil

import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA



# ----------------------------
# Fix random seed
# ----------------------------
RAND_SEED = 0
np.random.seed(RAND_SEED)


# ----------------------------
# Prepare data
# ----------------------------
def path_to_spector(file_path, sr=44100, channel=1, duration=4000):
    aud = AudioUtil.open(file_path)
    reaud = AudioUtil.resample(aud, sr)
    rechan = AudioUtil.rechannel(reaud, channel)
    dur_aud = AudioUtil.pad_trunc(rechan, duration)
    sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

    return sgram.numpy().reshape(-1)


# read data to construct train_set, we only use normal sound here
def get_X(data_dir):
    X  = []

    for file in glob.glob(os.path.join(data_dir, "*.wav")):
        if 'normal' in file:
            X.append(path_to_spector(file))
        else:  
            continue

    return np.array(X)


# read data to construct valid/test_set, we use both normal and void sound
def get_X_n_Y(data_dir):
    X, Y = [], []

    for file in glob.glob(os.path.join(data_dir, "*.wav")):
        if 'normal' in file:
            X.append(path_to_spector(file))
            Y.append(0)
        elif 'void' in file:
            X.append(path_to_spector(file))
            Y.append(1)
        else:  
            print("cannot identify file", file)

    return np.array(X), np.array(Y)


X_train = get_X('../train_set')
print("training set:", X_train.shape)

X_test, Y_test = get_X_n_Y('../test_set')
print("test set:", X_test.shape, Y_test.shape)


"""
    Training
"""

def score(a, b):
    #Normalized differnece of squares between a and b.
    loss = np.sum((a-b)**2, axis=1)
    print("avg reconstruction loss on the test set:", loss.mean())
    loss = (loss-loss.min())/(loss.max() - loss.min())
    return loss



# scale data to have zero mean and unit standard deviation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

    
### fit the model on the training set

N_COMP = 50  # this hyperparameter should be tuned based on the train set
print("N_COMP:", N_COMP)


"""
    three different PCA
"""
# auc: ~0.59
#pca = PCA(n_components=N_COMP, random_state=0)

# rbf: ~0.82, poly: ~0.72, linear: ~0.61
pca = KernelPCA(n_components=N_COMP, kernel='rbf', fit_inverse_transform=True, random_state=0)


X_train_feat = pca.fit_transform(X_train)

X_train_recon = pca.inverse_transform(X_train_feat)

recon_error = np.sum((X_train - X_train_recon)**2, axis=1).mean()

print("avg reconstruction loss on the training set:", recon_error)


### calculate roc on the test set

X_test_feat = pca.transform(X_test)
X_test_recon = pca.inverse_transform(X_test_feat)

test_score = score(X_test, X_test_recon)


fpr, tpr, _ = metrics.roc_curve(Y_test, test_score, pos_label=1)
auroc = metrics.auc(fpr, tpr)

print("auroc:", auroc)

plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)"%auroc)
plt.title('ROC Curve of KernelPCA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('KernelPCA_auc.pdf', bbox_inches='tight')
plt.show()

