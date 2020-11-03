## Learning Black-Box Attackers with Transferable Priors and Query Feedback

[Jiancheng Yang](https://jiancheng-yang.com/)\*, Yangzhou Jiang\*, [Xiaoyang Huang](http://scholar.google.com/citations?user=Svw7X6kAAAAJ&hl=en), [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ&hl=zh-CN), [Chenglong Zhao](https://scholar.google.com/citations?user=wl55lFoAAAAJ&hl=zh-CN).

*Neural Information Processing Systems (NeurIPS), 2020* ([arXiv](https://arxiv.org/abs/2010.11742))

### Abstract
This paper addresses the challenging black-box adversarial attack problem, where only classification confidence of a victim model is available. Inspired by consistency of visual saliency between different vision models, a surrogate model is expected to improve the attack performance via transferability. By combining transferability-based and query-based black-box attack, we propose a surprisingly simple baseline approach (named SimBA++) using the surrogate model, which significantly outperforms several state-of-the-art methods. Moreover, to efficiently utilize the query feedback, we update the surrogate model in a novel learning scheme, named High-Order Gradient Approximation (HOGA). By constructing a high-order gradient computation graph, we update the surrogate model to approximate the victim model in both forward and backward pass. The SimBA++ and HOGA result in Learnable Black-Box Attack (LeBA), which surpasses previous state of the art by large margins: the proposed LeBA reduces 34%-78% queries, while keeping higher attack success rates close to 100% in extensive ImageNet experiments, including attacking vision benchmarks and defensive models.

### Implementation
Due to the size limitation of supplementary, we do not provide weights of model and datasets here.  Note that our experiments setting is similar to paper [Improving Black-box Adversarial Attacks with a Transfer-based Prior](https://arxiv.org/pdf/1906.06919.pdf), including models and attack setting. Code and experiment setting will be open source after acceptance.

#### Requirements 
The dependent package we use include: `pytorch=1.2.0, pandas=0.25.1, pillow=5.4.1, opencv-python=4.1.1.26`.  
Note that the key package is pytorch.



#### Prepare Data
You can put ImageNet images and label file in dir `images` or try our example images in `images` first.  
Note that you can find dataset IMGN1 in our paper in
[Baidu driver](https://pan.baidu.com/s/1nt5guRByhu-hVo-98fj0SA) (Password：wawy) and  [Google Driver](https://drive.google.com/file/d/1wMpxCPfloy13UlYxhFxM_5fn7Rr2kEPm/view?usp=sharing).

#### Prepare Models
You can prepare your own model as victim model or surrogate model, and  modify function `get_models` in `data_utils.py`.  
We provide pretrain ImageNet model from torchvision, note that we test with pretrained models from Tensorflow-Slim in paper.

##### Pretrain models and model used in exps 

You can find the models we used in experiments and pretrained in this folder on [Baidu Wangpan](https://pan.baidu.com/s/1--gi2rJagnGZ3kcY5vQKQg)(Password:r4z6) and [Google Drive](https://drive.google.com/file/d/11DHdogeJbunThQMP8PgORO7kcOggPSD-/view?usp=sharing).



#### Script of Repo
LeBA2.py: Main script of LeBA attack, incluing 5 attack mode (train, test, SimBA, SimBA+, SimBA++).
run_attack.py:  Wrap script to run LeBA.
data_utils.py: Functions to provide data, models, and log class
get_result.py: Evaluate result file.
defense: Contain defense method, but currently only Jpeg Compression is provided.

#### Run LeBA
- RUN different (train, test, SimBA, SimBA+, SimBA++) mode in sequence with script  
Use run_attack.py, it will save all the result files to the dir like: 'note_log_all/inception_v3_Dec10_f1'. Please edit run_attack.py to specify the attack mode(train, test, SimBA, SimBA+, SimBA++), else it will run 5 attack mode in sequence.
```
python run_attack.py --gpu_id=0,1,2 --script=LeBA10.py --model1=inception_v3 --model2=resnet152
```
- To run SimBA+ mode  
```
python LeBA10.py --mode=simba+ --model1=inception_v3 --model2=resnet152 --input_dir=images --label=labels --out_dir="your output dir" 
```
- To run SimBA++ mode  
```
python LeBA10.py --mode=simba++ --model1=inception_v3 --model2=resnet152 --input_dir=images --label=labels --out_dir="your output dir" 
```

- To run LeBA   
```
python LeBA10.py --mode=train --model1=inception_v3 --model2=resnet152 --input_dir=images --label=labels --out_dir="your output dir" --pretrain_weight=""
```

- To run LeBA(test) mode  
After run LeBA, use weight trained in LeBA to test on another dataset.
```
python LeBA10.py --mode=test --model1=inception_v3 --model2=resnet152 --input_dir=imagesset2 --label=labels --out_dir="same output dir as train" --pretrain_weight="this_weight"
```




To attack defensive model:
```
#for Jpeg Compression
python run_attack.py --gpu_id=0,1,2 --script=LeBA10.py --model1=inception_v3 --model2=resnet152 --defense_method=jpeg
```

To evaluate results:
Modify `out_dir` in  `get_result.py`
and run `python get_result.py`
result will be save in result dir.




