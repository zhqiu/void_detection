"""
    generate normal/void sound based on existing data

    based on: https://github.com/ajc327/railway_defect_detection/blob/main/7.%20%20Synthesis%20of%20audio%20.ipynb
"""

import os
import shutil
import soundfile as sf
from tqdm import trange
import numpy as np
import librosa
import cv2


JAMES_CROSSLEY_FREQ_BANK = np.array([       # refer to Fig. 58~62 in James Crossley's report
    [106, 168, 193, 303, 495, 753],         # case 1
    [140, 168, 204, 305, 501, 756],         # case 2
    [168, 168, 200, 309, 498, 757],         # case 3
    [143, 168, 195, 313, 499, 756],         # case 4
    [106, 168, 194, 303, 493, 755]])        # case 5


"""
    generate normal sound by sampling
"""
def sample_normal(normal_sound, n_samples, sr, length, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)

    for i in trange(n_samples):
        loc = np.random.randint(0, len(normal_sound)-length)
        normal_slice = normal_sound[loc:loc+length]
        sf.write(os.path.join(save_path, "normal_"+str(i)+".wav"), normal_slice, sr, subtype='PCM_24')
   


"""
    generate void sound
"""
def find_matching_freqs(filter_bank, freqs):
    # This takes the target frequencies from the filter bank and finds the closest match 
    # from the available frequencies generated by the mel spectrogram
    # Point 3 on Page 21 in Cai's report
    out = []
    indices =[]
    for freq in filter_bank: 
        a = min(freqs, key=lambda x:abs(x-freq))
        out.append(a)
        indices.append(int(np.where(freqs==a)[0]))
    return out, indices


def gen_window(win_size, indices, filter_shape):
    # This generates a spectrogram slice using the given indices(where do we want the bright spot)
    # win_size defines the size of the slice. This should be (n_mels by desired length)
    # Point 4 on Page 21 in Cai's report
    out = np.zeros(win_size)
    n2 = win_size[1] // 2
    for i in indices:
        out[i, n2] = 4
    out = cv2.GaussianBlur(out, filter_shape, 0)
    return out 


def gen_feature(win_size, filter_shape, filter_bank, freqs, f_max):
    _, indx = find_matching_freqs(filter_bank, freqs)
    b = gen_window(win_size, indx, filter_shape)
    
    return librosa.feature.inverse.mel_to_audio(b, fmax=f_max)


def add_feature(x, feature, scale):
    x = librosa.util.normalize(x)
    feature = librosa.util.normalize(feature)* scale 
    
    loc = int((len(x)-len(feature))/2)+np.random.randint(-int(len(feature)/8),int(len(feature)/8))
    x[loc:loc+len(feature)] += feature

    return librosa.util.normalize(x)


def gen_void_samples(normal_sound, n_samples, sr, length, save_path, 
                     frequency_set=JAMES_CROSSLEY_FREQ_BANK,        
                     n_mels=128, f_max=1024,                         # size of Mel spectrogram
                     snr=1.0, wh=13, h=13, v=11,
                     cover_old=True):
    if os.path.isdir(save_path) and cover_old:
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    elif not os.path.isdir(save_path):
        os.mkdir(save_path)

    freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=f_max)       # create an empty Mel spectrogram
    
    for i in trange(n_samples):
        frequency_bank = frequency_set[np.random.randint(0, len(frequency_set))]

        loc = np.random.randint(0, len(normal_sound)-length)
        normal_sample = normal_sound[loc:loc+length]         # sample normal sound fist

        wh = np.random.choice([9,11,13])
        # generate void feature
        feature = gen_feature(win_size=(n_mels, wh), filter_shape=(h,v), filter_bank=frequency_bank, freqs=freqs, f_max=f_max)
        x = add_feature(normal_sample, feature, snr)
        sf.write(os.path.join(save_path, "void_snr_"+str(snr)+"_"+str(i)+".wav"), x, sr, subtype='PCM_24')



if __name__ == "__main__":
    normal_sound, normal_sr = librosa.load("../data/British-trains-at-High-Speed.wav")

    void_sound, void_sr = librosa.load("../data/void.wav")
    target_length = len(void_sound)

    N_train_samples = 1600
    N_val_samples = 200
    N_test_samples = 200

    # generate training/validation/test sets = 8:1:1

    # sample normal sound samples
    sample_normal(normal_sound, N_train_samples//2, normal_sr, target_length, save_path="../train_set")
    sample_normal(normal_sound, N_val_samples//2, normal_sr, target_length, save_path="../val_set")
    sample_normal(normal_sound, N_test_samples//2, normal_sr, target_length, save_path="../test_set")

    # generate void sound samples
    gen_void_samples(normal_sound, N_train_samples//4, normal_sr, target_length, snr=1.0, save_path="../train_set", cover_old=False)
    gen_void_samples(normal_sound, N_train_samples//4, normal_sr, target_length, snr=0.5, save_path="../train_set", cover_old=False)

    gen_void_samples(normal_sound, N_val_samples//4, normal_sr, target_length, snr=1.0, save_path="../val_set", cover_old=False)
    gen_void_samples(normal_sound, N_val_samples//4, normal_sr, target_length, snr=0.5, save_path="../val_set", cover_old=False)

    gen_void_samples(normal_sound, N_test_samples//4, normal_sr, target_length, snr=1.0, save_path="../test_set", cover_old=False)
    gen_void_samples(normal_sound, N_test_samples//4, normal_sr, target_length, snr=0.5, save_path="../test_set", cover_old=False)
