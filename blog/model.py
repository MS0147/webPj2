import cv2
import time,os,sys
import librosa, librosa.display 

from matplotlib import pyplot as plt

import time,os,sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Encoder,Decoder,AD_MODEL,weights_init,print_network
from metric import evaluate


dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))



##
class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features




##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i


class BeatGAN(AD_MODEL):


    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator( opt).to(device)
        self.G.apply(weights_init)
        if not self.opt.istest:
            print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        if not self.opt.istest:
            print_network(self.D)


        self.bce_criterion = nn.BCELoss().cuda()
        self.mse_criterion=nn.MSELoss().cuda()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label= 0


        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def set_input(self, input):
        #self.input.data.resize_(input[0].size()).copy_(input[0])
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
        #self.gt.data.resize_(input[1].size()).copy_(input[1])
        with torch.no_grad():
            self.gt.resize_(input[1].size()).copy_(input[1])

        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            #self.fixed_input.data.resize_(input[0].size()).copy_(input[0])
            with torch.no_grad():
                self.fixed_input.resize_(input[0].size()).copy_(input[0])


    ##
    def optimize(self):
        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()

    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.D(self.input)
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --

        #to~~~ delete if some problem will happen
        #print('self.out_d_real: ',type(self.out_d_real))
        #print('self.device: ', self.device)
        #print('self.batchsize: ', self.batchsize)
        #print('self.real_label: ',self.real_label)
        #print(self.real_label.device)
        #print('torch.full: ', torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor))
        
        
        #below the code, if .cuda() isn't exist, torch.full.device is cpu
        #print('torch.full is go to cpu?: ', torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor).cuda().device)
        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor).cuda())

        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.FloatTensor).cuda())


        self.err_d=self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')

    ##
    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input)


        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv=self.mse_criterion(self.feat_fake,self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x


        self.err_g =  self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()


    ##
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }


        return errors

        ##

    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]

        return  self.fixed_input.cpu().data.numpy(),fake.cpu().data.numpy()

    ##


    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_,y_pred=self.predict(self.dataloader["val"])
        rocprc,rocauc,best_th,best_f1=evaluate(y_,y_pred)
        return rocauc,best_th,best_f1

    def predict(self,dataloader_,scale=True, path="/"):
        with torch.no_grad():

            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)
                                        
            fake_path = path + "/fake/"
            real_path = path + "/real/"
            diff_path = path + "/diff/"
            
            if not os.path.exists(fake_path):
                os.makedirs(fake_path)
                
            if not os.path.exists(real_path):
                os.makedirs(real_path)
                
            if not os.path.exists(diff_path):
                os.makedirs(diff_path)
                
            fig1 = plt.figure(figsize=(5,5))
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)
                
                
                fake = self.fake.cpu().numpy()

                #plt.imshow(np.reshape(fake[1], [128,128,1]))
                
                img = librosa.display.specshow(fake[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
                plt.axis('off')
                fig1.savefig(fake_path+"fake"+str(i)+".png")
                fig1.savefig('static/imgFile/fake.png')

                real = self.input.cpu().numpy()
                img = librosa.display.specshow(real[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
                #fig1.axis('off')
                #plt.imshow(np.reshape(real[1], [128,128,1]))
                fig1.savefig(real_path+"real"+str(i)+".png")
                fig1.savefig('static/imgFile/real.png')
                
                
                # load images
                image1 = cv2.imread(real_path+"real"+str(i)+".png")
                image2 = cv2.imread(fake_path+"fake"+str(i)+".png")

                # compute diff
                difference = cv2.subtract(image2, image1)

                 # color the mask red
                Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                difference[mask != 255] = [0, 0, 255]

                 # add the red mask to the images to make the differences obvious
                image2[mask != 255] = [0, 0, 255]

                 # store images
                cv2.imwrite(diff_path+"diffOverImage"+str(i)+".png", image2)
                cv2.imwrite('static/imgFile/real.png', image2)

                # error = torch.mean(torch.pow((d_feat.view(self.input.shape[0],-1)-d_gen_feat.view(self.input.shape[0],-1)), 2), dim=1)
                #
                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_=self.gt_labels.cpu().numpy()
            y_pred=self.an_scores.cpu().numpy()

            return y_,y_pred


    def predict_for_right(self,dataloader_,min_score,max_score,threshold,save_dir):
        '''
        :param dataloader:
        :param min_score:
        :param max_score:
        :param threshold:
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair=[]
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)

                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))


                # # Save test images.

                batch_input = self.input.cpu().numpy()
                batch_output = self.fake.cpu().numpy()
                ano_score=error.cpu().numpy()
                assert batch_output.shape[0]==batch_input.shape[0]==ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair)>=100:
                        break
                    normal_score=(ano_score[idx]-min_score)/(max_score-min_score)
                    print('normal_score, threshold',normal_score, threshold)
                    if normal_score>=threshold:
                        test_pair.append((batch_input[idx],batch_output[idx]))

            # print(len(test_pair))
            self.saveTestPair(test_pair,save_dir)



    def test_type(self):
        self.G.eval()
        self.D.eval()
        #res_th=self.opt.threshold
        res_th=0.1695923507213
        print('threshold: ',res_th)
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False, path=os.path.join(save_dir, "N"))
        print('y_pred_N: ',y_pred_N)
        print('y_N: ',y_N)
        over_all=np.concatenate([y_pred_N])
        over_all_gt=np.concatenate([y_N])
        min_score,max_score=0.0005832395,0.010918294
        #save fig for Interpretable
        # self.predictForRight(self.dataloader["test_N"], save_dir=os.path.join(save_dir, "N"))
        self.predict_for_right(self.dataloader["test_N"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "N"))
        #aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print('finish')

    def test_time(self):
        self.G.eval()
        self.D.eval()
        size=self.dataloader["test_N"].dataset.__len__()
        start=time.time()

        for i, (data_x,data_y) in enumerate(self.dataloader["test_N"], 0):
            input_x=data_x
            for j in range(input_x.shape[0]):
                input_x_=input_x[j].view(1,input_x.shape[1],input_x.shape[2]).to(self.device)
                gen_x,_ = self.G(input_x_)

                error = torch.mean(
                    torch.pow((input_x_.view(input_x_.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end=time.time()
        print((end-start)/size)

