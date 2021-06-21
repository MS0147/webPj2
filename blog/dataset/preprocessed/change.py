
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display 
import cv2

n_data = np.load('N_samples.npy') 
s_data = np.load('S_samples.npy') 
v_data = np.load('V_samples.npy') 
f_data = np.load('F_samples.npy') 
q_data = np.load('Q_samples.npy')



n_fft_n= 256
win_length_n=64
hp_length_n=2
sr = 360 

data =n_data #데이터 종류

lst = [] #npy로 저장할 데이터들
length = len(data) #출력할 데이터 개수


for i in range(length):
    #원래 ECG 그래프 그리기
    #ax1 = fig1.add_subplot(length,2,2*(i+1)-1)
    #ax1.plot(data[i,0,:])
   
    # STFT 이미지 그리기
    #ax2 = fig1.add_subplot(length,2,2*(i+1))
             
    #STFT
    D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
    
    #ampiltude로 변환
    magnitude = np.abs(D_highres)
             
    #amplitude를 db 스케일로 변환
    log_spectrogram = librosa.amplitude_to_db(magnitude)
             
    #화이트 노이즈 제거
    log_spectrogram = log_spectrogram[:,10:150]
             
    #128,128로 resize
    log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
    
    #스펙트로그램 출력
    #img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hp_length_n, ax=ax2, y_axis="linear", x_axis="time")
             
    #컬러바
    #fig.colorbar(img, ax=ax2)# format="%+2.f dB")
    
    #print(log_spectrogram.shape)
    
    lst.append(log_spectrogram)
    if i%30==0:
        print(i,'/',length)

#npy로 저장 
lst = np.array(lst)
output_filename = 'n_spectrogram'
print(lst.shape)
np.save(output_filename, lst)


##########

data =s_data #데이터 종류

lst = [] #npy로 저장할 데이터들
length = len(data) #출력할 데이터 개수


for i in range(length):
    #원래 ECG 그래프 그리기
    #ax1 = fig1.add_subplot(length,2,2*(i+1)-1)
    #ax1.plot(data[i,0,:])
   
    # STFT 이미지 그리기
    #ax2 = fig1.add_subplot(length,2,2*(i+1))
             
    #STFT
    D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
    
    #ampiltude로 변환
    magnitude = np.abs(D_highres)
             
    #amplitude를 db 스케일로 변환
    log_spectrogram = librosa.amplitude_to_db(magnitude)
             
    #화이트 노이즈 제거
    log_spectrogram = log_spectrogram[:,10:150]
             
    #128,128로 resize
    log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
    
    #스펙트로그램 출력
    #img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hp_length_n, ax=ax2, y_axis="linear", x_axis="time")
             
    #컬러바
    #fig.colorbar(img, ax=ax2)# format="%+2.f dB")
    
    #print(log_spectrogram.shape)
    
    lst.append(log_spectrogram)
    if i%30==0:
        print(i,'/',length)

#npy로 저장 
lst = np.array(lst)
output_filename = 's_spectrogram'
print(lst.shape)
np.save(output_filename, lst)

##########

data =v_data #데이터 종류

lst = [] #npy로 저장할 데이터들
length = len(data) #출력할 데이터 개수


for i in range(length):
    #원래 ECG 그래프 그리기
    #ax1 = fig1.add_subplot(length,2,2*(i+1)-1)
    #ax1.plot(data[i,0,:])
   
    # STFT 이미지 그리기
    #ax2 = fig1.add_subplot(length,2,2*(i+1))
             
    #STFT
    D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
    
    #ampiltude로 변환
    magnitude = np.abs(D_highres)
             
    #amplitude를 db 스케일로 변환
    log_spectrogram = librosa.amplitude_to_db(magnitude)
             
    #화이트 노이즈 제거
    log_spectrogram = log_spectrogram[:,10:150]
             
    #128,128로 resize
    log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
    
    #스펙트로그램 출력
    #img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hp_length_n, ax=ax2, y_axis="linear", x_axis="time")
             
    #컬러바
    #fig.colorbar(img, ax=ax2)# format="%+2.f dB")
    
    #print(log_spectrogram.shape)
    
    lst.append(log_spectrogram)
    if i%30==0:
        print(i,'/',length)

#npy로 저장 
lst = np.array(lst)
output_filename = 'v_spectrogram'
print(lst.shape)
np.save(output_filename, lst)

##########

data =f_data #데이터 종류

lst = [] #npy로 저장할 데이터들
length = len(data) #출력할 데이터 개수


for i in range(length):
    #원래 ECG 그래프 그리기
    #ax1 = fig1.add_subplot(length,2,2*(i+1)-1)
    #ax1.plot(data[i,0,:])
   
    # STFT 이미지 그리기
    #ax2 = fig1.add_subplot(length,2,2*(i+1))
             
    #STFT
    D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
    
    #ampiltude로 변환
    magnitude = np.abs(D_highres)
             
    #amplitude를 db 스케일로 변환
    log_spectrogram = librosa.amplitude_to_db(magnitude)
             
    #화이트 노이즈 제거
    log_spectrogram = log_spectrogram[:,10:150]
             
    #128,128로 resize
    log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
    
    #스펙트로그램 출력
    #img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hp_length_n, ax=ax2, y_axis="linear", x_axis="time")
             
    #컬러바
    #fig.colorbar(img, ax=ax2)# format="%+2.f dB")
    
    #print(log_spectrogram.shape)
    
    lst.append(log_spectrogram)
    if i%30==0:
        print(i,'/',length)

#npy로 저장 
lst = np.array(lst)
output_filename = 'f_spectrogram'
print(lst.shape)
np.save(output_filename, lst)

##########

data =q_data #데이터 종류

lst = [] #npy로 저장할 데이터들
length = len(data) #출력할 데이터 개수


for i in range(length):
    #원래 ECG 그래프 그리기
    #ax1 = fig1.add_subplot(length,2,2*(i+1)-1)
    #ax1.plot(data[i,0,:])
   
    # STFT 이미지 그리기
    #ax2 = fig1.add_subplot(length,2,2*(i+1))
             
    #STFT
    D_highres = librosa.stft(data[i,0,:].flatten(), n_fft=n_fft_n, hop_length=hp_length_n, win_length=win_length_n)
    
    #ampiltude로 변환
    magnitude = np.abs(D_highres)
             
    #amplitude를 db 스케일로 변환
    log_spectrogram = librosa.amplitude_to_db(magnitude)
             
    #화이트 노이즈 제거
    log_spectrogram = log_spectrogram[:,10:150]
             
    #128,128로 resize
    log_spectrogram = cv2.resize(log_spectrogram, (128,128), interpolation = cv2.INTER_AREA)
    
    #스펙트로그램 출력
    #img = librosa.display.specshow(log_spectrogram, sr=sr, hop_length = hp_length_n, ax=ax2, y_axis="linear", x_axis="time")
             
    #컬러바
    #fig.colorbar(img, ax=ax2)# format="%+2.f dB")
    
    #print(log_spectrogram.shape)
    
    lst.append(log_spectrogram)
    if i%30==0:
        print(i,'/',length)

#npy로 저장 
lst = np.array(lst)
output_filename = 'q_spectrogram'
print(lst.shape)
np.save(output_filename, lst)
