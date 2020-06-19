import os
import numpy as np
import copy

def test_new(log_file, times=2501):
    
    with open(log_file,'r') as f:
        lines = f.readlines()
        L2_list=[]
        counts_list=[]
        success_list = []
        pic_list = []
        results={}
        i=0
        for line in lines:
            i+=1
            if i<0:
                continue
            if times>=1:
                if i==times:
                    break
            else:
                if i<times:
                    continue
            try:
                success = int(line.split()[1].split(':')[1].strip(','))
            except:
                print('line',line)
            counts = int(line.split()[2].split(':')[1].strip(','))
            L2 = float(line.split()[3].split(':')[1].strip(',').strip())
            if True:
                if counts<10000 and L2<16.38:
                    #counts_list.append(counts)
                    #L2_list.append(L2)
                    pass
                else:
                    #counts_list.append(10000)
                    #L2_list.append(16.37)
                    counts=10000
                    L2=16.38
                    success=0

                pic_name = line.split(' ')[0]
            results[pic_name] = [counts,L2,success]
            if pic_name not in results:# and success:
                counts_list.append(counts)
                L2_list.append(L2)
                success_list.append(success)
                pic_list.append(pic_name)
            else:
                pass
                #print(pic_name)

    print(np.array(L2_list).mean())    
    print(np.array(counts_list).mean())
    print(np.sum(np.array(counts_list)<300))
    print(np.array(success_list).sum()/len(success_list))
    print(len(counts_list))
    return pic_list, results


def test_old():
    out_dir='out'#_multi'
    with open(os.path.join(out_dir,"logging"),'r') as f:
        lines = f.readlines()
        L2_list=[]
        counts_list=[]
        success_list = []
        i=0
        for line in lines:
            i+=1
            if i==241:
                break
            print(line)
            success = int(line.split('(')[1].split(',')[0]=='True')
            if success:
                counts = int(line.split('counts:')[1].split(',')[0])
                L2 = float(line.split('L2:')[1].strip())
                if counts<4000:
                    counts_list.append(counts)
                    L2_list.append(L2)
            success_list.append(success)
            
    print(np.array(L2_list).mean())    
    print(np.array(counts_list).mean())
    print(np.array(success_list).sum()/len(success_list))



def test_P_RGF(input_dir,pic_list=None, results=None):
#    out_dir='../Prior-Guided-RGF/uniform_out3'
#    out_dir='../Prior-Guided-RGF/sss_newout'
    out_dir='../Prior-Guided-RGF/'+input_dir


 
    with open(os.path.join(out_dir,"temp_log"),'r') as f:
        lines = f.readlines()
        print("lines len", len(lines))
        L2_list=[]
        counts_list=[]
        success_list = []
        results={}
        for line in lines:
            pic_name = line.split()[0]
            if pic_list is not None and pic_name not in pic_list:
                continue
            try:
                counts = int(line.split('queries: ')[1].split(' ')[0])
                L2 = float(line.split('norm:')[1].strip())
                #print(results[pic_name], counts,L2)
                counts_list.append(counts)
                L2_list.append(L2)
            except:
                counts = 10000
                L2 = 16.37
                counts_list.append(counts)
                L2_list.append(L2) 
            success = int(line.split(' ')[1].split(' ')[0]=='succeed;')
            success_list.append(success)
            results[pic_name] = [counts,L2,success]
    print(np.array(L2_list).mean())    
    print(np.array(counts_list).mean())
    print(np.array(success_list).sum()/len(success_list))
    print('image num', len(L2_list))
    return results


if __name__=='__main__':
    out_dir = "note_log_all/inception_v3_Dec26_f8"
    log_file = os.path.join(out_dir,"log_train_images_idx0_0")
    pic_list,results2 = test_new(log_file,1630)
    print("1116, baseline版\n")
   
    out_dir = "note_log_all/inception_v3_Dec27_f10"
    log_file = os.path.join(out_dir,"log_train_images_idx0_0")
    pic_list,results = test_new(log_file,1630)
    print("1116, baseline版\n")

    out_dir = "note_log_all/1125_f0"
    log_file = os.path.join(out_dir,"log_train_dec30")
    pic_list,results = test_new(log_file,1630)
    print("1116, baseline版\n")


    counts_list=[]
    L2_list=[]
    success_list=[]
    for r in results:
        if r in results2:
            counts_list.append(results2[r][0])
            L2_list.append(results2[r][1])
            success_list.append(results2[r][2])
            if results2[r][2]==0:
            #print(r, results2[r][0])
                pass
        else:
            #print(r)
            pass
    print(len(L2_list))
    print(np.array(L2_list).mean()) 
    print(np.array(counts_list).mean())
    print(np.sum(np.array(counts_list)<300))
    print(np.array(success_list).sum()/len(success_list))    
    print()


    '''
    out_dir='all_outs/out_new3'
    log_file = os.path.join(out_dir,"logging")
    pic_list,results2 = test_new(log_file)
    new_results = copy.deepcopy(results2)
    counts_list=[]
    L2_list=[]
    for pic in results:
        if results[pic][0]<=5:
            new_results[pic][0]=1
            new_results[pic][1] = results[pic][1]
        L2_list.append(new_results[pic][1])
        counts_list.append(new_results[pic][0])
    #print(np.array(L2_list).mean())    
    #print(np.array(counts_list).mean())
    print()'''
    p_results = test_P_RGF("incept_resnet2_out")
    p_results = test_P_RGF("incept4_out")
    p_results = test_P_RGF("sss_newout")
    #
    #test_P_RGF()
    counts_list=[]
    L2_list=[]
    for pic in results:
        if results[pic][0]<=15:
            p_results[pic][0]=results[pic][0]
            p_results[pic][1] = results[pic][1]
        L2_list.append(p_results[pic][1])
        counts_list.append(p_results[pic][0])
    print()
    print(np.array(L2_list).mean())    
    print(np.array(counts_list).mean())
    print()
