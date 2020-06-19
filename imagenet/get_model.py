import torch
try:
    from . import models
except:
    import models
import os


root_dir = os.path.split(os.path.realpath(__file__))[0]



def get_model(model_name):
    model = getattr(models, model_name)(pretrained=True)
        #raise Exception("No model named %s"%net_name)
    return model

if __name__=='__main__':
    img = torch.rand([1,3,224,224])
    print(root_dir)
    for name in ['vgg16', 'resnet100', 'densenet100']:
        net = get_model('vgg16')
        out = net(img)
    print("Test Pass")

