## LeBA: Learning Black-Box Attacker via High-Order Gradient Approximation

We propose new black-box attack method, which combine transferability-based and query-based black-box attack.

### Abstract
This paper addresses the challenging black-boxadversarial attack problem, where only the classi-fication confidence of a victim model is available.Inspired by the consistency of visual saliency be-tween different vision models, a surrogate modelis expected to improve the attack performance via transferability. By combining the transferability-based and query-based black-box attack, we pro-pose  a  surprisingly  simple  baseline  approach(named  SimBA++)  using  the  surrogate  model,which significantly outperforms severalstate-of-the-artmethods. Moreover, to efficiently utilizethe query results, we update the surrogate modelin a novel learning scheme, named High-OrderGradient Approximation (HOGA). By construct-ing a high-order gradient computation graph, weupdate the surrogate model to approximate thevictim model in both forward and backward pass.The SimBA++ and HOGA result inLearnableBlack-BoxAttack (LeBA), which surpasses previ-ousstate of the artby large margins: the proposedLeBA reduces 44%-85% queries, while keepinghigher attack success rates close to 100% in ex-tensive experiments, including attacking visionbenchmarks and defensive models on ImageNet.

### Implementation
Due to the size limitation of supplementary, we do not provide weights of model and datasets here.  Note that our experiments setting is similar to paper [Improving Black-box Adversarial Attacks with a Transfer-based Prior](https://arxiv.org/pdf/1906.06919.pdf), including models and attack setting. Code and experiment setting will be open source after acceptance.

#### Prepare Data
You can put ImageNet images and label file in dir `images` or try our example images in `images` first.  
Note that you can find dataset IMGN1 in our paper from this [Repo](https://github.com/prior-guided-rgf/Prior-Guided-RGF). 

#### Prepare Models
You can prepare your own model as victim model or surrogate model, and  modify function `get_models` in `data_utils.py`.  
We provide pretrain ImageNet model from torchvision, note that we test in pretrain model from Tensorflow-Slim in paper.

#### Script of Repo
LeBA2.py: Main script of LeBA attack
run_attack.py:  Wrap script to run LeBA, set log dir
data_utils.py: Functions to provide data, models, and log class
get_result.py: Evaluate result file.
defense: Contain defense method, but currently only Jpeg Compression is provided.

#### Run LeBA

Use run_attack.py, it will save all the result files to dir like: 'note_log_all/inception_v3_Dec10_f1'
```
python run_attack.py --gpu_id=0,1,2 --script=LeBA2.py --model1=inception_v3 --model2=resnet152
```

To attack defensive model:
```
#for Jpeg Compression
python run_attack.py --gpu_id=0,1,2 --script=LeBA2.py --model1=inception_v3 --model2=resnet_v2_152 --defensive_model=jpeg
```

To evaluate results:
Modify `out_dir` in  `get_result.py`
and run `python get_result.py`
result will be save in result dir.




