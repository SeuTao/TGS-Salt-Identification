# Kaggle TGS Salt Identification Challenge 2018 4th place code
This is the source code for my part of the 4th place solution to the [TGS Salt Identification Challenge](https://www.kaggle.com/c/data-science-bowl-2017/) hosted by Kaggle.com. 

## Recent Update

**`2018.10.22`**: single model training code updated.

**`2018.10.20`**: We achieved the 4th place on  [Kaggle TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).

#### Dependencies
pytorch 0.3

## Solution Development
#### Single model design

- input: 101 random pad to 128*128, random LRflip;
- encoder: resnet34, se-resnext50, resnext101_ibna, se-resnet101, se-resnet152, se resnet154;
- decoder: scse, hypercolumn (not used in network with resnext101_ibna, se_resnext101 backbone), ibn block, dropout;
- Deep supervision structure with Lovasz softmax (a great idea from Heng);
We designed 6 single models for the final submission;


#### Single model performace
| single model（10fold 7cycle）           |valid LB| public LB| privare LB|
| ---------------- | ---- | ---- | ---- |
|model_50|0.873|0.873|0.891   |
|model_50_slim|0.871|0.872|0.891|
|model_101A|0.868|0.870|0.889    |
|model_101B|0.870|0.871|0.891    |
|model_152|0.868|0.869| 0.888    |
|model_154|0.869|0.871| 0.890    |

#### Model ensemble performace
| ensemble model（cycle voting）|public LB| privare LB|
| ---------------- | ---- | ----|
|50+50_slim|0.873|0.891|
|50+50_slim+101B|0.873|0.892|
|50+50_slim+101A|0.873|0.892|
|50+50_slim+101A+101B|0.874|0.892|
|50+50_slim+101A+101B+154|0.874|0.892|
|50+50_slim+101A+101B+152+154|0.874|0.892|

#### Post processing
According to the  2D and 3D jigsaw results (amazing ideas and great job from @CHAN), we applied around 10 handcraft rules that gave a 0.010~0.011 public LB boost and 0.001 private LB boost.

|model|public LB| privare LB|
| ---------------- | ---- | ----|
|50+50_slim+101A+101B with post processing|0.884|0.893|

#### Data distill (Pseudo Labeling)
We started to do this part since the middle of  the competetion. As Heng posts, pseudo labeling  is pretty tricky and has the risk of overfitting. I am not sure whether it would boost the private LB untill the result is published. I just post our results here, the implementation details will be updated. 

| model with datadistill|public LB| privare LB|
| ---------------- | ---- | ----|
|model_34|0.877|0.893|
|model_50|0.880|0.893|
|model_101|0.880|0.894|
|model 34+50+101|0.879|0.895|
|model_34 with post processing|0.885|0.893|
|model_50 with post processing|0.886|0.894|
|model_101 with post processing|0.886|0.895|
|model 34+50+101 with post processing (final sub)|0.887|0.895|

#### Training
Train model_34 
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=train --model=model_34 --model_name=model_34_try --train_fold_index=0
```
Test model_34
```
CUDA_VISIBLE_DEVICES=0 python main.py --mode=test --model=model_34 --model_name=model_34_try --train_fold_index=0
```

#### Suggestions.

## Refference











