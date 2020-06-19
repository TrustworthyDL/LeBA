## LeBA: Learning Black-Box Attackers with Transferable Priors and Query Feedback

We propose new black-box attack method, which combine transferability-based and query-based black-box attack.

### Abstract
This paper addresses the challenging black-box adversarial attack problem, where1only classification confidence of a victim model is available.  Inspired by consistency of visual saliency between different vision models, a surrogate model is expected to improve the attack performance via transferability. By combining transferability-based and query-based black-box attack, we propose a surprisingly simple baseline approach (named SimBA++) using the surrogate model, which significantly outperforms severalstate-of-the-artmethods. Moreover, to efficiently utilize the query feedback, we update the surrogate model in a novel learning scheme, named High-Order Gradient Approximation (HOGA). By constructing a high-order gradient computation graph, we update the surrogate model to approximate the victim model in both forward and backward pass. The SimBA++ and HOGA result inLearnableBlack-BoxAttack (LeBA), which surpasses previous state of the art by large margins: the proposed LeBA reduces 34%-78% queries, while keeping higher attack success rates close to 100% in extensive ImageNet experiments, including attacking vision benchmarks and defensive models

### Implementation
Due to the size limitation of supplementary, we do not provide weights of model and datasets here.  Note that our experiments setting is similar to paper [Improving Black-box Adversarial Attacks with a Transfer-based Prior](https://arxiv.org/pdf/1906.06919.pdf), including models and attack setting. Code and experiment setting will be open source after acceptance.

#### Prepare Data
You can put ImageNet images and label file in dir `images` or try our example images in `images` first.  
Note that you can find dataset IMGN1 in our paper from this [Repo](https://github.com/prior-guided-rgf/Prior-Guided-RGF). 

#### Prepare Models
You can prepare your own model as victim model or surrogate model, and  modify function `get_models` in `data_utils.py`.  
We provide pretrain ImageNet model from torchvision, note that we test in pretrain model from Tensorflow-Slim in paper.

#### Script of Repo
LeBA2.py: Main script of LeBA attack, incluing 5 attack mode (train, test, SimBA, SimBA+, SimBA++).
run_attack.py:  Wrap script to run LeBA.
data_utils.py: Functions to provide data, models, and log class
get_result.py: Evaluate result file.
defense: Contain defense method, but currently only Jpeg Compression is provided.

#### Run LeBA

Use run_attack.py, it will save all the result files to the dir like: 'note_log_all/inception_v3_Dec10_f1'. Please edit run_attack.py to specify the attack mode(train, test, SimBA, SimBA+, SimBA++), else it will run 5 attack mode in sequence.
```
python run_attack.py --gpu_id=0,1,2 --script=LeBA10.py --model1=inception_v3 --model2=resnet152
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




