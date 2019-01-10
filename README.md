# Kaggle TGS Salt Identification Challenge 2018 4th place code
This is the source code for my part of the 4th place solution to the [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge) hosted by Kaggle.com. 

![image](https://github.com/SeuTao/Kaggle_TGS2018_4th_solution/blob/master/png/tgs.png)

## Recent Update

**`2018.11.06`**: jigsaw python code，dirty code of handcraft rules and pseudo label training code updated.

**`2018.10.22`**: single model training code updated.

**`2018.10.20`**: We achieved the 4th place on  [Kaggle TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).

#### Dependencies
- opencv-python==3.4.2
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- torch==0.3.1
- torchvision==0.2.1


## Solution Development
#### Single model design

- input: 101 random pad to 128*128, random LRflip;
- encoder: resnet34, se-resnext50, resnext101_ibna, se-resnet101, se-resnet152, se resnet154;
- decoder: scse, hypercolumn (not used in network with resnext101_ibna, se_resnext101 backbone), ibn block, dropout;
- Deep supervision structure with Lovasz softmax;
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
According to the  2D and 3D jigsaw results, we applied around 10 handcraft rules that gave a 0.010~0.011 public LB boost and 0.001 private LB boost.

|model|public LB| privare LB|
| ---------------- | ---- | ----|
|50+50_slim+101A+101B with post processing|0.884|0.893|

#### Data distill (Pseudo Labeling)
We started to do this part since the middle of  the competetion. Pseudo labeling  is pretty tricky and has the risk of overfitting. I am not sure whether it would boost the private LB untill the result is published. I just post our results here, the implementation details will be updated. 
Steps (as the following flow chart shows):
  1. Grabing the pseudo labels provided by previous predict (with post processing).
  2. Randomly split the test set into two parts, one for training and the other for predicting.
  3. To prevent overfitting to pseudo labels, we randomly select images from training set or test set (one part) with same probability in each mini batch.
  4. Training the new dataset in three different networks with same steps as mentioned previously.
  5. Predicting the test set (the other part) by all three trained models and voting the result.
  6. Repeat step 3 to 5 except that in this time we change two test parts.
  
 ![image](https://github.com/SeuTao/Kaggle_TGS2018_4th_solution/blob/master/png/flow_chart.png)

| model with datadistill|public LB| privare LB|placement
| ---------------- | ---- | ----| ---|
|model_34|0.877|0.8931|8
|model_50|0.880|0.8939|8
|model_101|0.880|0.8946|7
|model 34+50+101|0.879|0.8947|6
|model_34 with post processing|0.885|0.8939|8
|model_50 with post processing|0.886|0.8948|5
|model_101 with post processing|0.886|0.8950|5
|model 34+50+101 with post processing (final sub)|0.887|0.8953|4


#### Data Setup
save the train mask images to disk
```
python prepare_data.py 
```

#### Single Model Training
train model_34 fold 0：
```
CUDA_VISIBLE_DEVICES=0 python train.py --mode=train --model=model_34 --model_name=model_34 --train_fold_index=0
```
predict model_34 all fold：
```
CUDA_VISIBLE_DEVICES=0 python predict.py --mode=InferModel10Fold --model=model_34 --model_name=model_34
```

#### Ensemble and Jigsaw Post-processing
After you predict all 6 single models 10 fold test csv，use this two command to perform majority voting and post-processing.

a) solve Jigsaw map (only need to run for one time)

```
python predict.py --mode=SolveJigsawPuzzles
```

b) ensemble 6 model all cycles and post-processing, 'model_name_list' is the list of signle model names you train with the command above
```
python predict.py --mode=EnsembleModels --model_name_list=model_50A,model_50A_slim,model_101A,model_101B,model_152,model_154 ----save_sub_name=6_model_ensemble.csv
```
You'll get ensemble sub '6_model_ensemble.csv' and ensembel+jigsaw sub '6_model_ensemble-vertical-empty-smooth.csv'

#### Pseudo label training
After you get ensemble+jigsaw results, use command below to train with pseudo label. We randomly split the test set into two parts. For each model, we train twice with 50% pseudo labels each.

train model_34 with 6model output pseudo label:

a) part0 fold 0

```
python train.py --mode=train --model=model_34 --model_name=model_34_pseudo_part0 --pseudo_csv=6_model_ensemble-vertical-empty-smooth.csv --pseudo_split=0 --train_fold_index=0
```

b) part1 fold 0
```
python train.py --mode=train --model=model_34 --model_name=model_34_pseudo_part1 --pseudo_csv=6_model_ensemble-vertical-empty-smooth.csv --pseudo_split=1 --train_fold_index=1
```

#### Final Ensemble

```
python predict.py --mode=EnsembleModels --model_name_list=model_34_pseudo_part0,model_34_pseudo_part1,model_50A_slim_pseudo_part0,model_50A_slim_pseudo_part1,model_101A_pseudo_part0,model_101A_pseudo_part1 ----save_sub_name=final_sub.csv
```

The "final_sub-vertical-empty-smooth.csv" is all you need.


## Reference
- https://arxiv.org/abs/1608.03983 LR schedule
- https://arxiv.org/abs/1803.02579 Squeeze and excitation
- https://arxiv.org/abs/1411.5752 Hypercolumns
- https://arxiv.org/abs/1705.08790 Lovasz
- https://arxiv.org/abs/1712.04440 Data distillation













