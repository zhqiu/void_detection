"""
    logistic regression and SVM
"""

import glob, os
from audio_utils import AudioUtil

import matplotlib.pyplot as plt

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC



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


def get_X_n_Y(data_dir):
    X, Y = [], []

    for file in glob.glob(os.path.join(data_dir, "*.wav")):
        if 'normal' in file:
            X.append(path_to_spector(file))
            Y.append(1)
        elif 'void' in file:
            X.append(path_to_spector(file))
            Y.append(0)
        else:  
            print("cannot identify file", file)

    return np.array(X), np.array(Y)

X_train, Y_train = get_X_n_Y('../train_set')
print("training set:", X_train.shape, Y_train.shape)

X_test, Y_test = get_X_n_Y('../test_set')
print("test set:", X_test.shape, Y_test.shape)


"""
    Training
"""

for method in ['LogisticRegression', 'AdaBoostClassifier', 'SVM']:

    print("method:", method)

    if method == 'LogisticRegression':
        clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
    elif method == 'AdaBoostClassifier':
        clf = AdaBoostClassifier(random_state=0).fit(X_train, Y_train)
    elif method == 'SVM':
        clf = SVC(probability=True, random_state=0).fit(X_train, Y_train)


    mean_acc = clf.score(X_test, Y_test)
    print("Mean ACC on test set: %0.4f"%mean_acc)


    probs_void = clf.predict_proba(X_test)[:, 0]


    fpr, tpr, _ = metrics.roc_curve(Y_test, probs_void, pos_label=0)
    auroc = metrics.auc(fpr, tpr)

    print("auroc:", auroc)

    plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)"%auroc)
    plt.title('ROC Curve of '+method)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(method+'_auc.pdf', bbox_inches='tight')
    plt.show()



