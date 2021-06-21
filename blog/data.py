import random
import os
import numpy as np
import torch
from  torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import librosa
import matplotlib.pyplot as plt
import librosa, librosa.display 
import cv2

np.random.seed(42)

def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

def load_data(opt):
    train_dataset=None
    test_dataset=None
    val_dataset=None
    test_N_dataset=None
    test_S_dataset = None
    test_V_dataset = None
    test_F_dataset = None
    test_Q_dataset = None
    
    if opt.dataset=="ecg":
        ran_num=random.randint(0,100)
        n_data = np.load(os.path.join(opt.dataroot,'N_samples.npy'))[ran_num,0]
        #n_data = np.load('N_samples.npy')[0,0]
        n_fft_n= 256
        win_length_n=64
        hp_length_n=2
        sr = 360 
        
        D_highres = librosa.stft(n_data.flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
        
        magnitude = np.abs(D_highres)
             
        #amplitude를 db 스케일로 변환
        log_spectrogram = librosa.amplitude_to_db(magnitude)

        #화이트 노이즈 제거
        log_spectrogram = log_spectrogram[:,10:150]

        #128,128로 resize
        log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
        
        N_samples = log_spectrogram.reshape(1,1,128,128)
        
        for i in range(N_samples.shape[0]):
            N_samples[i] = normalize(N_samples[i])
        
        test_N,test_N_y=N_samples, np.ones((N_samples.shape[0], 1))
        
        test_N_dataset = TensorDataset(torch.Tensor(test_N), torch.Tensor(test_N_y))

    # assert (train_dataset is not None  and test_dataset is not None and val_dataset is not None)

    dataloader = {
                    "test_N":DataLoader(
                            dataset=test_N_dataset,  # torch TensorDataset format
                            batch_size=1,  # mini batch size
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False),
                    }
    return dataloader


def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y

def get_full_data(dataloader):

    full_data_x=[]
    full_data_y=[]
    for batch_data in dataloader:
        batch_x,batch_y=batch_data[0],batch_data[1]
        batch_x=batch_x.numpy()
        batch_y=batch_y.numpy()

        # print(batch_x.shape)
        # assert False
        for i in range(batch_x.shape[0]):
            full_data_x.append(batch_x[i,0,:])
            full_data_y.append(batch_y[i])

    full_data_x=np.array(full_data_x)
    full_data_y=np.array(full_data_y)
    assert full_data_x.shape[0]==full_data_y.shape[0]
    print("full data size:{}".format(full_data_x.shape))
    return full_data_x,full_data_y



