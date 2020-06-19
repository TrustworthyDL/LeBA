#from .sample import inception,model
import torch
from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image
import torch.nn.functional as F
import numpy as np
import random
from scipy import ndimage
from skimage.io import imread, imsave
#from .var_min import *
#from .feature_squeeze import *
#from .GD.nips_deploy.v3 import get_model
import torch.nn as nn

def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    return im

def jpeg_compression_batch(imgs):
    jpeg_imgs = torch.zeros(imgs.shape).to(imgs.device)
    for i in range(imgs.shape[0]):
        jpeg_imgs[i] = _jpeg_compression(imgs[i].cpu()).to(imgs.device)
    return jpeg_imgs

def padding_layer_iyswim(rescaled, rnd, high):
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint( 0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded

def gauss_filter(A, r):
    output_imgs = np.zeros((A.shape), dtype=np.float32)
    for i in range(A.shape[0]):
        for j in range(3):
            output_imgs[i, j, :, :] = ndimage.filters.gaussian_filter(A[i, j, :, :], r)
    return output_imgs


def get_defense(defense_method,net=None):
    if defense_method=='jpeg':
        def jpeg_defense(x, preprocess):
            jpeg_x = jpeg_compression_batch(x)
            return net( preprocess( jpeg_x ) )
        return jpeg_defense
    '''
    elif defense_method=='GD':
        config, net = get_model()
        state_dict= torch.load('defense/denoise_incepv3_012.ckpt')['state_dict']
        #for key in state_dict:
        #    print(key)
        net.load_state_dict(state_dict)
        net1 = nn.DataParallel(net.net.cuda(), device_ids=[i for i in range(2)])
        net1.eval()
        def GD_query(x, preprocess):
            return net1(preprocess(x), True)[-1]
        return GD_query
    elif defense_method=='random':
        def randomization(x,preprocess):
            out = 0
            for j in range(30): 
                resize_shape = np.random.randint(300, 331)
                resize_x = F.interpolate(x, resize_shape)
                padded_input = padding_layer_iyswim(resize_x, resize_shape, 331)
                out+=net(preprocess(padded_input))
            return out/30.0
        return randomization
    elif defense_method=='median_filter':
        def median_filter(x,preprocess):
            median_x = ndimage.filters.median_filter(x.detach().cpu().numpy(), size=[1,1,3,3],mode='reflect')
            median_x = torch.tensor(median_x).cuda()
            return net( preprocess(median_x ) )
        return median_filter
    elif defense_method=='gaussian_filter':
        def gaussian_filter(x, preprocess):
            gauss_x = gauss_filter(x.detach().cpu().numpy()*255.0, 2)/255.0
            gauss_x = torch.tensor(gauss_x).cuda()
            return net( preprocess(gauss_x) )
        return gaussian_filter
    elif defense_method=='var_min':
        VarMin = TotalVarMin()
        def var_min(x, preprocess):
            var_x, _ = VarMin(x.cpu().numpy())
            var_x = torch.tensor(var_x).cuda()
            return net( preprocess(var_x) )
        return var_min
    elif defense_method=='fs':
        FeatureSqueeze = FeatureSqueezing([0,1])
        def feature_squeeze(x, preprocess):
            f_x, _ = FeatureSqueeze(x.cpu().numpy())
            f_x = torch.tensor(f_x).cuda()
            return net( preprocess(f_x) )
        return feature_squeeze'''

    


