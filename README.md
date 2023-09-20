# hydro-learn: *An intelligent package for solving hydrogeology engineering issues*


##  Overview

*Hydro-learn* is a Python-based package for solving hydro-geology engineering issues. From methodologies based on 
Machine Learning,It brings novel approaches  for reducing numerous losses during the hydrogeological  
exploration projects. It allows to: 
- reduce the cost of permeability coefficient (k) data collection during the engineering projects,
- Guide drillers for to locating the drilling operations, 
- predict the water content in the well such as the level of water inrush, ...

## Licence 

*WATex* is under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) License. 

## Installation 

The system requires preferably Python 3.10+. 

## Demos 

### Predict permeability coefficient ``K`` from logging dataset using MXS approach
 
MXS stands for mixture learning strategy. It uses upstream unsupervised learning for 
``k`` -aquifer similarity label prediction and the supervising learning for 
final ``k``-value prediction. For our toy example, we use two boreholes data 
stored in the software and merge them to compose a unique dataset. In addition, we dropped the 
``remark`` observation which is subjective data not useful for ``k`` prediction as:

```python

import hlearn
h= hlearn.fetch_data("hlogs", key='h502 h2601', drop_observations =True ) # returns log data object.
h.feature_names
Out[3]: Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter'],
          dtype='object')
hdata = h.frame 
```
``k`` is collected as continue values (m/darcies) and should be categorized for the 
naive group of aquifer prediction (NGA). The latter is used to predict 
upstream the  MXS target ``ymxs``.  Here, we used the default categorization 
provided by the software and we assume that in the area, there are at least ``2`` 
groups of the aquifer. The code is given as: 
```python 
mxs = hlearn.MXS (kname ='k', n_groups =2).fit(hdata) 
ymxs=mxs.predictNGA().makeyMXS(categorize_k=True, default_func=True)
mxs.yNGA_ [62:74]
Out[4]: array([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
ymxs[62:74]
Out[5]: array([ 0,  0,  0,  0, 12, 12, 12, 12, 12, 12, 12, 12])
```
To understand the transformation from NGA to MXS target (``ymxs``), please, have a look 
of the following [paper](http://dx.doi.org/10.2139/ssrn.4326365).
Once the MXS target is predicted, we call the ``make_naive_pipe`` function to 
impute, scale, and transform the predictor ``X`` at once into a compressed sparse 
matrix ready for final prediction using the [support vector machines](https://ieeexplore.ieee.org/document/708428) and 
[random forest](https://www.ibm.com/topics/random-forest) as examples. Here we go: 
```python 
X= hdata [h.feature_names]
Xtransf = hlearn.make_naive_pipe (X, transform=True) 
Xtransf 
Out[6]: 
<218x46 sparse matrix of type '<class 'numpy.float64'>'
	with 2616 stored elements in Compressed Sparse Row format> 
Xtrain, Xtest, ytrain, ytest = hlearn.sklearn.train_test_split (Xtransf, ymxs ) 
ypred_k_svc= hlearn.sklearn.SVC().fit(Xtrain, ytrain).predict(Xtest)
ypred_k_rf = hlearn.sklearn.RandomForestClassifier ().fit(Xtrain, ytrain).predict(Xtest)
```
We can now check the ``k`` prediction scores using ``accuracy_score`` function as: 
```python 
hlearn.sklearn.accuracy_score (ytest, ypred_k_svc)
Out[7]: 0.9272727272727272
hlearn.sklearn.accuracy_score (ytest, ypred_k_rf)
Out[8]: 0.9636363636363636
```
As we can see, the results of ``k`` prediction are quite satisfactory for our 
toy example using only two boreholes data. Note that things can become more 
interesting when using many boreholes data. For more in 
depth, visit our [examples page](https://watex.readthedocs.io/en/latest/glr_examples/index.html). 


## Contributions 

1. Department of Geophysics, School of Geosciences & Info-physics, [Central South University](https://en.csu.edu.cn/), China.
2. Hunan Key Laboratory of Nonferrous Resources and Geological Hazards Exploration Changsha, Hunan, China
3. Laboratoire de Geologie Ressources Minerales et Energetiques, UFR des Sciences de la Terre et des Ressources Mini�res, [Universit� F�lix Houphou�t-Boigny]( https://www.univ-fhb.edu.ci/index.php/ufr-strm/), Cote d'Ivoire.

Developer: [_L. Kouadio_](https://wegeophysics.github.io/) <<etanoyau@gmail.com>>