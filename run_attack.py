import subprocess
import os
import time

print('Current run file',__file__)
def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#script = "saba_attack_exp.py"
def get_cmd(task_id, gpu_id, mode, model1, model2,input_dir, label_file, out_dir,pretrain_weight, seed=0):
    FL_rate=args.FL_rate
    cmd_temp = '''python %s --task_id=%d --gpu_id=%s --mode=%s --model1=%s --model2=%s  \
            --input_dir=%s --label_file=%s --out_dir=%s --pretrain_weight=%s --seed=%d --FL_rate=%s \
            --defense_method=%s --lr=%s --ba_num=%d --ba_interval=%d
            '''% (script, task_id, gpu_id, mode,  model1, model2, input_dir, label_file,
                 out_dir, pretrain_weight, seed, FL_rate,  args.defense_method, args.lr, args.ba_num, args.ba_interval)
    return cmd_temp

def run_cmd(cmd, out_dir_al, type_):
    local_time = time.strftime("%b%d_%H%M%S", time.localtime()) 
    log_name = out_dir_al+'/slog/%s_'%type_+local_time
    with open(log_name+'.err','w') as f1:
        with open(log_name+'.out','w') as f2:
            subprocess.run(cmd, shell=True, stderr=f1, stdout=f2)
            
import argparse

parser = argparse.ArgumentParser(description='RUN BA&SA L3 Query Attack')
parser.add_argument('--task_id',default=1, help='task idx for log dir name', type=int)
#parser.add_argument('--input_dir',default='./images', type=str)
parser.add_argument('--model1', default='inception_v3', help='victim model name', type=str)
parser.add_argument('--model2', default='resnet152', help='surrogate model name', type=str)
parser.add_argument('--gpu_id', default='0,1,2', type=str)
#parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--FL_rate', default=0.01, help='index control forward loss', type=float)
parser.add_argument('--lr', default=0.005, help='learning rate', type=float)
parser.add_argument('--script', default="LeBA10.py", type=str)
parser.add_argument('--extra_name', default="", help='extra name for log dir', type=str)
parser.add_argument('--defense_method', default="", type=str)
parser.add_argument('--ba_num', default=10, help='TIMI iteration num', type=int)
parser.add_argument('--ba_interval', default=20, help='TIMI interval num', type=int)
parser.add_argument('--out_root', default='note_log_all', help='log root dir', type=str)






#parser.add_argument('--out_dir', default='incept3_1122_f2', type=str)



args = parser.parse_args()   

script = args.script

gpu_id=args.gpu_id
task_id=args.task_id
model1 = args.model1

out_dir = args.model1+'_'+args.extra_name+time.strftime("%b%d", time.localtime())+'_%d'%(task_id)
out_root= args.out_root
out_dir_al = out_root+'/'+out_dir
#pretrain_weight = "../sss_query_attack/note_log_all/mgpu02_lr0.001_3_n55f3/snapshot/resnet152_v2_final.pth"
pretrain_weight = ""
check_mkdir(out_dir_al+'/slog')
local_time = time.strftime("%b%d_%H%M%S", time.localtime()) 
model2 = args.model2
subprocess.run('cp %s note_log_all/%s/'%(script,out_dir), shell=True)
subprocess.run('cp %s note_log_all/%s/'%(__file__, out_dir), shell=True)

for idx in range(1):
    seed = idx
    cmd = get_cmd(idx, gpu_id, 'train', model1, model2, "./images",'labels', out_dir_al, pretrain_weight, seed)
    run_cmd(cmd, out_dir_al, 'train_images')
    
    cmd = get_cmd(idx, gpu_id, 'test', model1,  model2, "./images",'labels',out_dir_al, 'this_weight', seed)
    run_cmd(cmd, out_dir_al,'test_images')

    cmd = get_cmd(idx, gpu_id, 'SimBA++', model1, model2,  "./images",'labels', out_dir_al, '', seed)
    run_cmd(cmd, out_dir_al,'SimBA++_images')
    
    cmd = get_cmd(idx, gpu_id, 'SimBA+', model1, model2,  "./images",'labels', out_dir_al, '', seed)
    run_cmd(cmd, out_dir_al,'SimBA+_images')
    
    cmd = get_cmd(idx, gpu_id, 'SimBA', model1, model2,  "./images",'labels', out_dir_al, '', seed)
    run_cmd(cmd, out_dir_al,'SimBA_images')


