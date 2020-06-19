from test_tool import test_new
import torch
from collections import OrderedDict
import os


out_dir = 'note_log_all/vgg16_Dec26_f8'

with open(out_dir+'/test2.res','w') as f1:
    for idx in range(3):
        result = OrderedDict()
        for ty in os.listdir(out_dir):
            print(ty)
            #f = 'log_inception_v3_'+ty+'_idx%d_0'%idx
            f = ty
            #pic_list, res = test_new(out_dir+'/'+f)
            try:
                pic_list, res = test_new(out_dir+'/'+f,2001)
            except:
                continue
            counts = [res[r][0] for r in res]
            L2 =  [res[r][1] for r in res]
            success = [res[r][2] for r in res]
            result[ty] = [torch.tensor(counts).float().mean(), torch.tensor(L2).mean(), torch.tensor(success).float().mean(),len(L2)]
            print()
        f1.write('\n'+out_dir+'_idx%d\n'%idx)
        for r in result:
            f1.write('%s: Counts:%f, L2:%f, success:%f, Num:%d\n'%(r, result[r][0], result[r][1], result[r][2],result[r][3]))
            pass

