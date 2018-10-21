# Kaggle TGS Salt Identification Challenge 2018 4th place code
This is the source code for my part of the 4th place solution to the [TGS Salt Identification Challenge](https://www.kaggle.com/c/data-science-bowl-2017/) hosted by Kaggle.com. 

#### Dependencies & data

#### General

## Recent Update

**`2018.10.20`**: code to be upload.

**`2018.10.20`**: We achieved the 4th place on  [Kaggle TGS Salt Identification Challenge](https://www.kaggle.com/c/data-science-bowl-2017/).


#### Single model performace
| single model（10fold）           |valid LB| public LB| privare LB|
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
We designed around 10 handcraft rules that gave a 0.010~0.011 public LB boost and 0.001 private LB boost.

#### Data distill

#### Training neural nets

#### Predicting neural nets

#### Training of submissions, combining submissions for final  submission.

#### Bugs and suggestions.











