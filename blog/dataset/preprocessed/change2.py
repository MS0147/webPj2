import numpy as np

N_samples = np.load('n_spectrogram.npy') 
S_samples = np.load('s_spectrogram.npy') 
V_samples = np.load('v_spectrogram.npy') 
F_samples = np.load('f_spectrogram.npy') 
Q_samples = np.load('q_spectrogram.npy')
##########

S_samples = S_samples.reshape(S_samples.shape[0],1,S_samples.shape[1],S_samples.shape[2])
V_samples = V_samples.reshape(V_samples.shape[0],1,V_samples.shape[1],V_samples.shape[2])
F_samples = F_samples.reshape(F_samples.shape[0],1,F_samples.shape[1],F_samples.shape[2])
Q_samples = Q_samples.reshape(Q_samples.shape[0],1,Q_samples.shape[1],Q_samples.shape[2])
N_samples = N_samples.reshape(N_samples.shape[0],1,N_samples.shape[1],N_samples.shape[2])

np.save('q_spectrogram', Q_samples)
np.save('v_spectrogram', V_samples)
np.save('s_spectrogram', S_samples)
np.save('f_spectrogram', F_samples)
np.save('n_spectrogram', N_samples)
