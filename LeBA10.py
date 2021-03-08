# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.optim as optim
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from data_utils import *
#from defense import *
#from task import imagenet
import cv2
from skimage.io import imread, imsave
from torch.distributions import Categorical
import torch.autograd as autograd
from defense.defense import get_defense 
import random

class QueryModel():
    '''  Query Model Class
    Args: defense_method (str): defense name, 
          model (nn.module): basic victim model 
    '''
    def __init__(self, defense_method='', model=None):
        if defense_method!='':
            # If using defensive method, Get defense net 
            self.defense_net = get_defense(args.defense_method, model)
        else:
            self.model = model
        self.defense_method=defense_method

    def get_query(self, out, labels):
        #return query results: score, cw loss and cross_entropy_loss
        if out.shape[1]==1001:
            c_labels = labels.clone()
        elif out.shape[1]==1000:
            c_labels = labels.clone() - 1
        with torch.no_grad():
            prob = F.softmax(out,dim=1)
            loss = nn.CrossEntropyLoss(reduction='none')(out, c_labels)
        score = prob.gather(1, c_labels.reshape([-1,1]))
        correct = prob.argmax(dim=1)==c_labels
        top2 = prob.topk(2)
        delta_score = torch.log(top2.values[:,0])-torch.log(top2.values[:,1])
        return score, delta_score, loss, correct

    def query(self, imgs, model, preprocess, labels):   
        # Query for no defense case
        with torch.no_grad():
            out = model(preprocess(imgs))
            return self.get_query(out,labels)

    def __call__(self, imgs,  preprocess, labels):
        if self.defense_method=='':
            return self.query(imgs, self.model, preprocess, labels)
        elif self.defense_method in ['jpeg', 'GD']:
            with torch.no_grad():
                out = self.defense_net(imgs, preprocess)
            return self.get_query(out,labels)
        else:
            raise NameError('False defense method')


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel.astype(np.float32)

def gauss_conv(img, k_size):
    kernel = gkern(k_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    stack_kernel = torch.Tensor(stack_kernel).to(device)
    out = F.conv2d(img, stack_kernel, padding=(k_size-1)//2, groups=3)
    return out

def distance(imgs1, imgs2=None, norm=2):
    #Compute L2 or L_inf distance between imgs1 and imgs2
    if imgs1.dim()==3:
        imgs1 = imgs1.unsqueeze(0)
        imgs2 = imgs2.unsqueeze(0)
    img_num = imgs1.shape[0]
    if imgs2 is None:
        if norm==2:
            distance = (imgs1.view([img_num,-1])).norm(2,dim=1)
            return distance   
    if norm==2:
        try:
            distance = (imgs1.view([img_num,-1])-imgs2.view([img_num,-1])).norm(2, dim=1)
        except:
            print(img_num, imgs1.shape, imgs2.shape)
    elif norm=='inf':
        distance = (imgs1.view([img_num,-1])-imgs2.view([img_num,-1])).norm(float('inf'), dim=1)
    return distance  

def update_img(imgs, raw_imgs, diff, max_distance):
    #update imgs: clip(imgs+diff), clip new_imgs to constrain the noise within max_distace
    if imgs.dim()==3:
        imgs = imgs.unsqueeze(0)
        raw_imgs = raw_imgs.unsqueeze(0)
        diff = diff.unsqueeze(0)
    diff_norm = distance( torch.clamp(imgs+diff,0,1), raw_imgs)
    factor =  (max_distance / diff_norm).clamp(0,1.0).reshape((-1,1,1,1))
    adv_diff = (torch.clamp(imgs+diff,0,1) - raw_imgs)*factor 
    adv_imgs = torch.clamp(raw_imgs+adv_diff,0,1)
    return adv_imgs

def normalize(input_):
    return input_ / input_.view([input_.shape[0],-1]).pow(2).mean(-1).sqrt().view([-1,1,1,1]).clamp(1e-12,1e6)

def update_slice(value, slice1, slice2, target):
    temp = value[slice1]
    temp[slice2] = target
    value[slice1] = temp



'''def get_diff(select, reference):
    diff_map = torch.zeros(reference.shape).to(device)
    diff = torch.zeros(select.shape[0]).to(device)
    for i in range(select.shape[0]):
        diff_map[i, select[i,0, 0], select[i,0,1], select[i,0,2]] = reference[i, select[i,0, 0], select[i,0,1], select[i,0,2]]
        diff[i] =  reference[i, select[i,0, 0], select[i,0,1], select[i,0,2]]
    return diff_map,diff'''

def get_gauss_diff(shape, select, k_size, epsilon=1.0):
    diff = torch.zeros([shape[0],shape[1],shape[2]+k_size-1,shape[3]+k_size-1])#.to(device)
    diff_kernel = torch.zeros([shape[0],k_size,k_size])
    for i in range(shape[0]):
        gauss_kernel = torch.tensor(torch.tensor(gkern(k_size,3))*epsilon)#.to(device)
        diff_kernel[i] = gauss_kernel  + torch.randn(gauss_kernel.shape)*gauss_kernel*0.1
        diff[i, select[i,0], select[i,1]:select[i,1]+k_size, select[i,2]:select[i,2]+k_size] += diff_kernel[i]                             
    if k_size!=1:
        diff = diff[:,:,k_size//2:-(k_size//2), k_size//2:-(k_size//2)]
    return diff,diff_kernel

def get_diff_gauss(selects, shape, reference,k_size):
    #Return Gaussian diff 
    diff,diff_kernel = get_gauss_diff(shape, selects[:,0,:], k_size)
    diff = diff.to(device)
    for i in range(diff.shape[0]):
        diff[i] = diff[i]/diff[i].max()
        diff[i]*=reference[i]
        diff_kernel[i] = diff_kernel[i]/diff_kernel[i].max()
        diff_kernel[i]*=reference[i]
    return diff,diff_kernel

def sample_byprob( probs, shape):
    #Sample one pixel per image according to probs
    with torch.no_grad():
        m = Categorical(probs)
        select = m.sample()
        c = select//(shape[2]*shape[3])
        w = select % shape[3]
        h = (select-c*shape[2]*shape[3])//shape[3]
        select = torch.stack([c,h,w]).transpose(1,0).long()
    return select

def select_points(mode='by_prob', probs=None, select_num=1):
    #Args: mode: 'by_prob': select pixel by prob map
    #		 	or 'max': select top k prob pixel
    # Sample Multi pixels.
    shape = probs.shape
    if mode=='by_prob':
        probs = probs.reshape([probs.shape[0],-1])
        selects = []
        for n in range(select_num):
            select = sample_byprob(probs,shape)
            selects.append(select)
        selects = torch.stack(selects).permute(1,0,2)
    elif mode=='max':
        probs = probs.reshape([probs.shape[0],-1])
        a, select = torch.topk(probs, select_num, dim=-1)
        c = select//(shape[2]*shape[3])
        w = select % shape[3]
        h = (select-c*shape[2]*shape[3])//shape[3]
        selects = torch.stack([c,h,w]).permute([1,2,0]).long() 
    return selects

def attack_black(images, labels, model, model2, preprocess1, preprocess2, counts, correct, last_query):
    ''' Black-box attack TIMI, run in the first iteration in LeBA Attack
        Args: preprocess1, preprocess2: preprocess function for model, model2
              counts, correct, last_query: Init records
    '''
    raw_imgs = images
    adv_img = images.clone()
    adv_img.requires_grad=True
    diff=0
    momentum=0.9
    epsilon= args.max_distance/16.37
    max_distance = args.max_distance
    img_num = images.shape[0]
    best_advimg = images.clone()
    def proj(imgs,diff, index, mask=None):
        return update_img(imgs, raw_imgs[index], diff, max_distance)
    
    for it in range(10):
        out = model2(preprocess2(adv_img))
        if out.dim()==1:
            out = out.unsqueeze(0)
        if out.shape[1]==1001:
            c_labels = labels
        elif out.shape[1]==1000:
            c_labels = labels-1
        loss = nn.CrossEntropyLoss()(out,c_labels)
        loss.backward()
        grad = adv_img.grad.data
        grad = gauss_conv(grad,9)
        diff_norm = (diff*momentum + grad).view(img_num,-1).norm(2,dim=1).clamp(1e-12, 1e12).reshape([img_num,1,1,1])
        diff = epsilon*(diff*momentum + grad)/diff_norm
        adv_img.data[correct] = proj(adv_img.data[correct], diff[correct], correct)
        adv_img.grad.zero_()
        model2.zero_grad()
        if it>2 and it%1==0: # TIMI in first iteration will query model during iterations, 
                             # it will early stop some query success sample, and won't update some no improve perturbation
            c1 = correct.clone()
            score1, q1, loss1, c1[correct] = query(adv_img.data[correct], preprocess1, labels[correct])
            counts[correct] +=1
            update_index = (q1<last_query[correct]).reshape([-1]) |(~c1[correct])
            update_slice(last_query, correct, update_index, q1[update_index])
            update_slice(best_advimg, correct, update_index, adv_img.data[correct][update_index])
            correct *=c1
            if correct.sum()==0:
                break
    adv_img = adv_img.detach()
    adv_img.requires_grad=False
    log.print('black_attack,distance: ',end='')
    log.print(distance(images, best_advimg))
    return best_advimg, adv_img



def get_trans_advimg(imgs, model2, labels, raw_imgs, ba_num):
    # TIMI for following iterations in LeBA, similar to attack_black function, but it won't query victim model during iteration
    # Args: ba_num: iteration num in TIMI
    adv_img = imgs.detach().clone()
    adv_img.requires_grad=True
    diff=0
    momentum = 0.9
    epsilon = args.max_distance/16.37
    max_distance = args.max_distance
    img_num = imgs.shape[0]
    def proj(img,diff, mask=None):
        return update_img(img, raw_imgs, diff, max_distance)
    for i in range(ba_num):   
        out = model2(preprocess2(adv_img))
        if out.dim()==1:
            out = out.unsqueeze(0)
        if out.shape[1]==1001:
            c_labels = labels
        elif out.shape[1]==1000:
            c_labels = labels-1
        loss = nn.CrossEntropyLoss()(out,c_labels)
        loss.backward()
        grad = adv_img.grad.data
        grad = gauss_conv(grad,9)
        diff_norm = (diff*momentum + grad).view(img_num,-1).norm(2,dim=1).clamp(1e-8, 1e8).reshape([img_num,1,1,1])
        diff = epsilon*(diff*momentum + grad)/diff_norm
        adv_img.data = proj(adv_img.data, diff)
        adv_img.grad.zero_()
        model2.zero_grad()
    adv_img.requires_grad=False
    return adv_img.detach()


def adjust_learning_rate(optimizer,lr):
    #lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class TrainModelS():
    #HOGA: Class Method to train surrogate model 
    def __init__(self):
        self.train_num=0
        self.lamda_dict={}
        self.d_loss_record={}
        self.s_loss_record={}
        self.d_loss_sum=0
        self.s_loss_sum=0
        
    def get_lamda(self, filenames):
        #Adaptive gamma in paper, Here is to get adaptive lamda
        for filename in filenames:
            if filename in self.d_loss_record and filename in self.s_loss_record:
                d_loss_list = self.d_loss_record[filename]
                s_loss_list = self.s_loss_record[filename]
                if self.train_num>50:
                    lamda2 = self.s_loss_sum / self.d_loss_sum    # Use history s_loss sum and d_loss sum, compute lamda2
                    self.lamda_dict[filename] = self.lamda_dict[filename]*0.9 + lamda2*0.1   #Update lamda with lamda2 using momentum 
                else:
                    self.lamda_dict[filename] = 3.0
            else:
                self.lamda_dict[filename] = 3.0

    def __call__(self, filenames, imgs, model2, labels, diff, query_score, query_loss, last_loss, optimizer):
        #Call HOGA, train model2 
        '''Args:
            diff: Current query perturbation
            query_score: Current query score with (imgs+diff)
            query_loss: Current query loss with (imgs+diff)
            last_loss: History query loss with (imgs)
            model2: surrogate model
            optimizer: optimizer for model2
        '''
        self.get_lamda(filenames)
        lamda = torch.tensor([self.lamda_dict[filename] for filename in filenames]).to(device)
        self.train_num+=1
        d_loss = query_loss-last_loss    #Get Query delta loss
        adv_imgs = imgs.detach().clone()
        adv_imgs.requires_grad=True
        out = model2(preprocess2(adv_imgs))
       # print(out.shape,labels.shape)
        if out.dim()==1:
            out = out.unsqueeze(0)
        if out.shape[1]==1001:
            c_labels = labels
        elif out.shape[1]==1000:
            c_labels = labels-1
        prob = F.softmax(out,dim=1)
        s_score = prob.gather(1, c_labels.reshape([-1,1]))
        loss = nn.CrossEntropyLoss(reduction='none')(out, c_labels)   #Note that using cross entropy loss to train surrogate model here
        grad = autograd.grad(loss.sum(), adv_imgs,create_graph = True)   # Create High Order Compute Graph
        grad = grad[0]
        s_loss = (diff.detach()*grad).view([imgs.shape[0],-1]).sum(dim=1)  #diff*s_grad: surrogate model loss with diff.
        forward_loss = nn.MSELoss()(s_score, query_score.detach())    #Forward Loss: approximate forward-pass score number 
        backward_loss = nn.MSELoss()(s_loss/lamda, d_loss.detach())   
                    #Backward Loss: Minimize difference between surrogate model loss and query loss. equal to high-order gradient approximation.
        loss2 =backward_loss + forward_loss*args.FL_rate

        model2.zero_grad()
        loss2.backward()
        optimizer.step()
        model2.zero_grad()
        optimizer.zero_grad()
        del adv_imgs
        with open(train_log_file,'a')  as f:
            for i in range(s_loss.shape[0]):
                f.write("(%f,%f,%f), "%(s_loss[i], d_loss[i],lamda[i]))
            f.write('\n')
        for i in range(len(filenames)):
            filename = filenames[i]
            if filename not in self.d_loss_record:
                self.d_loss_record[filename] = []
            if filename not in self.s_loss_record:
                self.s_loss_record[filename] = []  
            self.d_loss_record[filename].append(d_loss[i].detach().cpu())
            self.s_loss_record[filename].append(s_loss[i].detach().cpu())
            self.d_loss_sum+=d_loss[i].detach().cpu().abs()
            self.s_loss_sum+=s_loss[i].detach().cpu().abs()


def get_data(data_iter, num):
    #Get Data from data_loader
    #Args:
    #	num: get data number.
    filenames = []
    imgs = []
    labels = []
    for i in range(num):
        try:
            data = next(data_iter)
            imgs.append(data['image'].to(device))
            labels.append(data['label'].to(device))
            filenames.append(data['filename'][0])
            #data_end=False
        except:
            log.print("Data Iterater finished")
            break
    return imgs, labels, filenames



def before_query_iter(imgs, labels,  model, model2, preprocess1, preprocess2, with_TIMI, with_s_prior, log):
    #First iteration in LeBA
    raw_imgs = imgs.clone()
    #First query victim model. 
    #Get last_score, last_query(cw_loss:delta log score for simbda) and last_loss(cross entropy loss for TIMI), 
    #correct:correctly classified: Not correct = Success
    last_score, last_query, last_loss, correct = query(imgs,  preprocess1, labels)
    _, a, b, correct_s = query2(imgs, model2, preprocess2, labels)
    log.print("Init correct rate, model %f, model_s %f"%(correct.float().mean(), correct_s.float().mean()))
    img_num = imgs.shape[0]
    counts = torch.ones([img_num]).to(device)
    end_type = torch.zeros([img_num]).to(device)
    prior_prob = torch.ones(imgs.shape).to(device)
    if correct.sum()>0:
        #RUN TIMI, and update counts, correct, last_query status
        if with_s_prior:
            best_advimg, adv_img = attack_black(imgs, labels, model, model2, preprocess1, preprocess2, counts, correct, last_query)
            #Update prior prob according to  accumulative gradient in TIMI, accumulative gradient is more stable.
            prior_prob = (best_advimg-raw_imgs).abs().clamp(1e-6,1e6)  #修改： best_advimg to adv_img
            prior_norm = prior_prob.view(img_num,-1).norm(2,dim=1).clamp(1e-12, 1e12).reshape([img_num,1,1,1])
            prior_prob = prior_prob/prior_norm 
        if with_TIMI and with_s_prior:
            imgs = best_advimg
    last_score,last_query, last_loss, correct = query(imgs,  preprocess1, labels)
    counts+=correct.float()
    end_type[~correct] = 1
    return imgs, counts, last_score, last_query, last_loss, correct,prior_prob, end_type

def index_(list1, index):
    new_list = []
    for i in range(index.shape[0]):
        if index[i].data==True:
            new_list.append(list1[i])
    return new_list

def normalizer(tensor):
    img_num = tensor.shape[0]
    norm = tensor.view(img_num,-1).norm(2,dim=1).clamp(1e-12, 1e12).reshape([img_num,1,1,1])
    return tensor/norm

def run_attack_train(model, model2, data_loader, minibatch,
                        preprocess1, preprocess2, log, optimizer, log_name,
                        if_train=True, with_TIMI=True, with_s_prior=True):
    '''
    Main function to run LeBA algorithm.
    We use batch for attack, and to accelerate speed, we introduce pipeline attack
    Pipeline attack means if one image has been breached, we add a new image to attack.
    Args:
        model: victim model
        model2: surrogate model
        data_loader: iterator return data 
        minibatch: batch size for attack
        preprocess1: Preprocess function for model1
        preprocess2: Preprocess functin for model2
        log: attack log class
        optimizer:  optimizer for model2(srrogate model)
        log_name: name of result file
        if_train: Flag of if train surrogate model, if 'if_train' off, function degrade to SimBA++
    '''
    data_iter = iter(data_loader)
    img_nums = len(data_loader)

    minibatch = minibatch if minibatch<=img_nums else img_nums
    correct_all = torch.ones([img_nums]).bool().to(device)  #record all correct(not success) flag
    counts_all = torch.zeros([img_nums]).to(device)         #Record all query numbers 
    end_type_all = torch.zeros([img_nums]).to(device).float()  #for debug
    L2_all = torch.zeros([img_nums]).to(device)         # Record final perturbation amount
    it=0
    img_id=0
    indices=torch.zeros([img_nums]).bool().to(device)   #Record indices of all has been attacked images
    indices[:minibatch] = True
    
    correct = torch.zeros([minibatch]).bool().to(device)  #Minibatch correct(not success) flag
    counts = torch.zeros([minibatch]).to(device)      # Record minibatch query numbers 
    end_type = torch.zeros([minibatch]).to(device).float()  #for debug
    max_query=10000        # max query budget
    epsilon= args.epsilon     #epsilon for SimBA part
    max_distance = args.max_distance #Max perturb budget  (L2 distance)
    b_num=0    
    #data_end=False
    get_new_flag = False
    def proj(imgs,diff, raw_imgs):  #Clip function 
        return update_img(imgs, raw_imgs, diff, max_distance)
    while True:          
        it+=1
        if it%50==1 or get_new_flag:  # Per 50 iteration, add new input data, and save success samples.
            get_new_flag = False 
            b_num+=1
            if b_num!=1:
                L2 = distance(imgs, raw_imgs)
                end_type_all[indices] = end_type
                L2_all[indices] = L2
                with open(out_dir+'/'+log_name,'a')  as f:
                    for i in range(len(imgs)):
                        if correct[i]==False or counts[i]>max_query:
                            #Write attack result to result file
                            f.write(filenames[i]+' Success:%d'%(~correct[i])+' counts:%d, L2:%.5f, end_type:%d \n'%(counts[i], L2[i], end_type[i]))
                            adv_img = imgs[i].cpu().detach().numpy().clip(0, 1).transpose((1,2,0))
                            imsave(out_dir+'/images/'+filenames[i], adv_img)  #Save adversarial example
                            correct[i]=False  
            correct_all[indices] = correct
            counts_all[indices] = counts
            if img_id==img_nums and correct.sum()==0 and get_new_flag==False: #Attack finish
                break
            if correct.sum()<minibatch:
                indices *=correct_all
                new_imgs, new_labels,  new_filenames = get_data(data_iter, minibatch-(correct).sum()) #Get new data to attack
                get_new = (new_labels!=[])  #New attack is available
                if get_new:
                    new_labels = torch.cat(new_labels) 
                    indices[img_id:img_id+new_labels.shape[0]] = True
                    img_id+=new_labels.shape[0]
                    new_raw_imgs = torch.cat(new_imgs).clone()
                    #Run TIMI first
                    #Get new_imgs and several update properties
                    new_imgs, counts0, last_score0, last_query0, last_loss0, correct0, prior_prob0, end_type0 = \
                                        before_query_iter(torch.cat(new_imgs), new_labels,  model, model2,preprocess1, preprocess2, with_TIMI, with_s_prior, log)
                    
                    last_improve0 = torch.zeros([new_imgs.shape[0]]).to(device)
     
                if b_num==1:
                    correct=correct0
                #Update all the propertities in pipeline
                last_score = last_score0 if b_num==1 else torch.cat([last_score[correct], last_score0]) if get_new else last_score[correct]
                last_query = last_query0 if b_num==1 else torch.cat([last_query[correct], last_query0]) if get_new else last_query[correct]
                last_loss = last_loss0 if b_num==1 else  torch.cat([last_loss[correct], last_loss0]) if get_new else last_loss[correct] 
                imgs =  new_imgs if b_num==1 else torch.cat([imgs[correct], new_imgs]) if get_new else imgs[correct]
                raw_imgs =  new_raw_imgs if b_num==1 else torch.cat([raw_imgs[correct], new_raw_imgs]) if get_new else raw_imgs[correct]
                
                filenames = new_filenames if b_num==1 else index_(filenames,correct) + new_filenames if get_new else index_(filenames,correct)
                labels = new_labels if b_num==1 else torch.cat([labels[correct], new_labels]) if get_new else labels[correct]
                prior_prob = prior_prob0 if b_num==1 else torch.cat([prior_prob[correct], prior_prob0])  if get_new else prior_prob[correct]
                counts = counts0 if b_num==1 else torch.cat([counts[correct],counts0]).to(device) if get_new else counts[correct]
                end_type = end_type0 if b_num==1 else torch.cat([end_type[correct],end_type0]).to(device) if get_new else end_type[correct]
                last_improve = last_improve0 if b_num==1 else torch.cat([last_improve[correct],last_improve0]).to(device) if get_new else last_improve[correct]
                correct = correct0 if b_num==1 else torch.cat([correct[correct],correct0]).to(device) if get_new else correct[correct]
                print(b_num, correct)
                print("Init last_query:", last_query)
                log.print(filenames)
        
        if it%args.ba_interval==(args.ba_interval-1) and with_s_prior:
            #Run TIMI 
            adv_imgs = get_trans_advimg(imgs[correct], model2, labels[correct], raw_imgs[correct],args.ba_num)
            score3, d_score3, loss3, c3 = query(adv_imgs,  preprocess1, labels[correct])
            #Update prior_prob
            prior_prob[correct] =normalizer((adv_imgs-raw_imgs[correct]).abs().clamp(1e-6,1e6)) #+ torch.rand(imgs[correct].shape).to(device)*0.2
            update_index = (d_score3<last_query[correct]) | (~c3) #| ((last_query[correct]==1.0) & (last_improve[correct]>=80)) 
            # If TIMI attack improve query result(cw_loss: delta log score), update images and properties.
            if update_index.sum()>0:
                new_prior = (adv_imgs-imgs[correct])[update_index]
                if with_TIMI:
                    update_slice(imgs, correct, update_index, adv_imgs[update_index])
                    update_slice(last_score, correct, update_index, score3[update_index])
                    update_slice(last_query, correct, update_index, d_score3[update_index])
                    update_slice(last_loss, correct, update_index, loss3[update_index])
            counts+=correct.float()  # update counts record
            correct[correct]*=c3  #update correct flags
            end_type[(end_type==0)*(~correct)] = 2
            if correct.sum()==0:
                get_new_flag=True
                continue 
        if it%10==0: # log
            score, d_score, loss, c = query(imgs, preprocess1, labels)  #(Only for log)
            L2 = distance(imgs, raw_imgs)
            log.print('It%d, Query:%d, d_score:%f, loss1:%f,  correct: %f, L2: %.4f'%(it, counts.mean(), last_query.mean(), last_loss.mean(), correct.float().mean(), L2.mean()))
            logs_str="Counts: "
            logs_L2="L2: "
            logs_score="score: "
            logs_loss="loss: "
            for i in range(imgs.shape[0]):
                logs_str+="%d, "%counts[i]
                logs_L2+="%.3f, "%L2[i]
                logs_score+="%.3f, "%last_query[i]
                logs_loss+="%.3f, "%last_loss[i]
            log.print(logs_str)
            log.print(logs_L2)
            log.print(logs_score)
        
        #Run SimBA+: 
        reference = torch.ones(imgs.shape[0])*epsilon
        if not with_s_prior:
            prior_prob = torch.ones(imgs.shape).to(device)
        selects = select_points(mode='by_prob', probs=prior_prob, select_num=1) #Select point according to prior prob got by TIMI.
        k_size = int( (args.max_distance*25/16.38 +1)//2*2+1 )
        diff,diff_kernel = get_diff_gauss(selects, imgs.shape, reference, k_size=k_size)   #Add gaussian noise on select pixel.

        c1 = correct.clone()      
        adv_imgs = proj(imgs[correct], diff[correct], raw_imgs[correct])     
        score1, d_score1, loss1, c1[correct] = query(adv_imgs,  preprocess1, labels[correct]) #Query model1 with +diff noise
        update_index = (d_score1<last_query[correct]) | (~c1[correct])  
        if if_train: #Use query information to train surrogate model (HOGA)
            train_model_s(index_(filenames,correct), imgs[correct], model2, labels[correct], adv_imgs-imgs[correct], score1, loss1, last_loss[correct], optimizer)
        
        last_improve[correct]+=1
        #If query result improve update imgs and properties
        update_slice(imgs, correct, update_index, adv_imgs[update_index])
        update_slice(last_score, correct, update_index, score1[update_index])
        update_slice(last_query, correct, update_index, d_score1[update_index])
        update_slice(last_loss, correct, update_index, loss1[update_index])
        update_slice(last_improve, correct, update_index, 0)
        counts+=correct.float()
        #record not correct and not update with +diff indices
        remain = correct.clone() 
        update_slice(remain, correct, update_index, False)
        correct*=c1
        end_type[(end_type==0)*(~correct)] = 3
        if correct.sum()==0:
            get_new_flag=True
            continue   
        if remain.sum()>0:  #For not correct and not update with +diff samples
            c2 = correct.clone()
            adv_imgs = proj(imgs[remain], -diff[remain], raw_imgs[remain])  #Query model1 with -diff noise
            score2, d_score2, loss2, c2[remain] = query(adv_imgs,  preprocess1, labels[remain])
            if if_train:  #HOGA
                train_model_s(index_(filenames,remain), imgs[remain], model2, labels[remain], adv_imgs-imgs[remain], score2, loss2, last_loss[remain], optimizer)
            counts+=remain.float()
            update_index2 = (d_score2<last_query[remain]) | (~c2[remain])

            #If query result improve update imgs and properties
            last_improve[remain]+=1
            update_slice(imgs, remain, update_index2, adv_imgs[update_index2])
            update_slice(last_score, remain, update_index2, score2[update_index2])
            update_slice(last_query, remain, update_index2, d_score2[update_index2])
            update_slice(last_loss, remain, update_index2, loss2[update_index2])
            update_slice(last_improve, remain, update_index2, 0)
            correct*=c2
            end_type[(end_type==0)*(~correct)] = 3
        #score, d_score, loss, c = query(imgs, preprocess1, labels)
        
        if correct.sum()==0:
            get_new_flag=True
            continue 
        
    if if_train:  #Save train weight of surrogate model
        torch.save(model2.state_dict(),out_dir+'/snapshot/%s_final.pth'%args.model2)
    return counts_all, correct_all, end_type_all, L2_all
                



def parse_args():
    parser = argparse.ArgumentParser(description='BA&SA L3 Query Attack')
    parser.add_argument('--task_id',default=0, help='task id for log dir name', type=int)   
    parser.add_argument('--input_dir',default='./images', help='input dir of images', type=str)
    parser.add_argument('--label_file',default='old_labels', help='label file name in input dir', type=str)
    parser.add_argument('--model1',default='inception_v3', help="Name of victim Model", type=str)
    parser.add_argument('--model2',default='resnet152', help="Name of substitute Model",type=str)
    parser.add_argument('--gpu_id', default="0,1,2", help='using gpu id', type=str)
    parser.add_argument('--epsilon', default=0.1, help="Epsilon in Simba Attack part", type=float)
    parser.add_argument('--seed', default=1, help="Random number generate seed", type=int)
    parser.add_argument('--lr', default=0.005, help="Learning rate for train s_model.", type=float)
    parser.add_argument('--FL_rate', default=0.01, help="rate for forward loss", type=float)
    parser.add_argument('--defense_method', default='',  help="jpeg or GD supported for defense name", type=str)
    parser.add_argument('--pretrain_weight',default='', help="pretrained weight path for surrogate model", type=str)
    parser.add_argument('--mode', default="train", help="train(LeBA) / test(LeBA test mode(SimBA++)) / SimBA++ / SimBA+ / SimBA", type=str)
    parser.add_argument('--batch_size', default=0, help="batch_size, if = 0, compute batch_size with gpu number", type=int)
    parser.add_argument('--ba_num', default=10, help="iterations for TIMI attack", type=int)
    parser.add_argument('--ba_interval', default=20, help="interval for TIMI attack", type=int)
    parser.add_argument('--max_distance', default=16.37, help="max perturbation (L2 norm)", type=float)
    parser.add_argument('--out_dir', default='out', help="output dir", type=str)
    return parser.parse_args()    

args =  parse_args()

#Set random seed 
seed=args.seed
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed = seed


#Get victim model(model1) and surrogate model(model2) and wrap them for multi gpus.
cpu_model = get_model(args.model1)
cpu_model2 = get_model(args.model2)

data_loader = load_images_data(args.input_dir, 1, False, args.label_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
gpu_num = len(args.gpu_id.split(',')) 
if gpu_num==0:
    gpu_num=1
device = torch.device("cuda")
model = cpu_model.to(device)
model2 = cpu_model2.to(device)
model = nn.DataParallel(cpu_model.to(device),device_ids=[i for i in range(gpu_num)])
model2 = nn.DataParallel(cpu_model2.to(device), device_ids=[i for i in range(gpu_num)])

model.eval()
model2.eval()

#Set output dir 
out_dir = args.out_dir  #  used to be try20
check_mkdir(out_dir+'/images')
check_mkdir(out_dir+'/snapshot')
check_mkdir(out_dir+'/logs')
check_mkdir(out_dir+'/gauss_images')

#preprocess functions for model1, model2
preprocess1 = get_preprocess(args.model1)
preprocess2 = get_preprocess(args.model2)

#query functions for model1, model2
query = QueryModel(args.defense_method, model)
query2 = QueryModel('', model2).query

optimizer=0
b=0
log = Logger(out_dir+'/logs/') #log file
train_log_file = out_dir+'/train_log0'

log.print("Args:")
log.print(args)

if_train=False
with_TIMI = True
with_s_prior = True

if args.mode=='train': #LeBA
    if_train=True
    minibatch = gpu_num *8
elif args.mode=='test':  #LeBA test mode
    minibatch = gpu_num *8 
    if args.pretrain_weight=='':
        args.pretrain_weight=='this_weight'
elif args.mode=='SimBA++':  #SimBA++
    minibatch = gpu_num *8
    args.pretrain_weight = ''
elif args.mode=='SimBA+':
    minibatch = gpu_num *8
    args.pretrain_weight = ''
    with_TIMI = False
elif args.mode=='SimBA':
    minibatch = gpu_num *16
    with_TIMI = False
    with_s_prior = False


if args.batch_size!=0:
    minibatch = args.batch_size

if args.mode[:3]!='all':
    log_name = "log_"+args.mode+'_'+args.input_dir.split('/')[-1]+'_idx%d_0'%(args.task_id)  #result file name

    if args.pretrain_weight=='this_weight':  #Load last trained surrogate weight
        model2.load_state_dict(torch.load(args.out_dir+'/snapshot/'+args.model2+'_final.pth'))
    elif args.pretrain_weight!='':
        model2.load_state_dict(torch.load(args.pretrain_weight))

    data_iter = iter(data_loader)
    optimizer = optim.SGD(model2.parameters(), lr = args.lr, momentum=0.9)
    train_model_s = TrainModelS()  #function for train surrogate model
    
    #Run LeBA
    counts_all, correct_all,  end_type_all, L2_all = run_attack_train(model, model2, data_loader, minibatch, 
                    preprocess1, preprocess2, log, optimizer, log_name, if_train, with_TIMI, with_s_prior)
