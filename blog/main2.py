import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from options import Options

from data import load_data

# from dcgan import DCGAN as myModel


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
print('device: ',device)

opt = Options().parse()
print('hello')
print(opt)
dataloader=load_data(opt)
print("load data success!!!")

if opt.model == "beatgan":
    from model import BeatGAN as MyModel

else:
    raise Exception("no this model :{}".format(opt.model))


model=MyModel(opt,dataloader,device)
#model=MyModel(*args, **kwargs)
model.load_state_dict(torch.load('/output/beatgan/ecg/model/beatgan_folder_0_D.pkl'))
print('finish')
