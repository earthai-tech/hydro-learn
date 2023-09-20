# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio 

"""
load different data as a function 
=================================

Inspired from the machine learning popular dataset loading 

Created on Thu Oct 13 16:26:47 2022
@author: Daniel
"""
import scipy 
import numpy as np
from importlib import resources 
import pandas as pd 

from .io import _to_dataframe, DMODULE, description_loader, DESCR 
from ..utils.baseutils import read_data   
from ..utils.mlutils import split_train_test_by_id, existfeatures
from ..utils.funcutils import ( 
    to_numeric_dtypes , 
    smart_format,
    key_checker, 
    random_sampling,
    assert_ratio, 
    )
from ..utils.box import Boxspace

__all__= [ "load_hlogs","load_nlogs", "load_mxs"]


def load_hlogs (
        *,  return_X_y=False, as_frame =False, key =None,  split_X_y=False, 
        test_size =.3 , tag =None, tnames = None , data_names=None, 
         **kws): 
    
    drop_observations =kws.pop("drop_observations", False)
    
    cf = as_frame 
    key = key or 'h502' 
    # assertion error if key does not exist. 
    available_sets = {
        "h502", 
        "h2601", 
        'h1102',
        'h1104',
        'h1405',
        'h1602',
        'h2003',
        'h2602',
        'h604',
        'h803',
        'h805'
        }
    is_keys = set ( list(available_sets) + ["*"])

    key = key_checker(key, is_keys)
    
    data_file ='h.h5'
    with resources.path (DMODULE , data_file) as p : 
        data_file = p 
    if key =='*': 
        key = available_sets
        
    if isinstance (key, str): 
        data = pd.read_hdf(data_file, key = key)
    else: 
        data =  pd.concat( [ pd.read_hdf(data_file, key = k) for k in key ]) 

    if drop_observations: 
        data.drop (columns = "remark", inplace = True )
        
    frame = None
    feature_names = list(data.columns [:12] ) 
    target_columns = list(data.columns [12:])
    
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    try : 
        existfeatures(data[target_columns] , tnames)
    except Exception as error: 
        # get valueError message and replace Feature by target 
        msg = (". Acceptable target values are:"
               f"{smart_format(target_columns, 'or')}")
        raise ValueError(str(error).replace(
            'Features'.replace('s', ''), 'Target(s)')+msg)
        
    if  ( 
            split_X_y
            or return_X_y
            ) : 
        as_frame =True 
        
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = tnames, 
            target=data[tnames].values 
            )
        frame = to_numeric_dtypes(frame)
        
    if split_X_y: 
        X, Xt = split_train_test_by_id (data = frame , test_ratio= test_size, 
                                        keep_colindex= False )
        y = X[tnames] 
        X.drop(columns =target_columns, inplace =True)
        yt = Xt[tnames]
        Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if cf else (
            X.values, Xt.values, y.values , yt.values )
    
    if return_X_y: 
        data , target = data.values, target.values
        
    if ( 
            return_X_y 
            or cf
            ) : return data, target 
    
    return Boxspace(
        data=data.values,
        target=data[tnames].values,
        frame=data,
        tnames=tnames,
        target_names = target_columns,
        # XXX Add description 
        DESCR= '', # fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )

load_hlogs.__doc__="""\
Load the hydro-logging dataset.

Dataset contains multi-target and can be used for a classification or 
regression problem.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a Bowlspace object. See
    below for more information about the `data` and `target` object.
    .. versionadded:: 0.1.2
    
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.
    .. versionadded:: 0.1.3
split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) with the according to the test size ratio.  
test_size: float, default is {{.3}} i.e. 30% (X, y)
    The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
    respectively. 
tnames: str, optional 
    the name of the target to retreive. If ``None`` the full target columns 
    are collected and compose a multioutput `y`. For a singular classification 
    or regression problem, it is recommended to indicate the name of the target 
    that is needed for the learning task. 
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~watex.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
    
key: str, default='h502'
    Kind of logging data to fetch. Can also be the borehole ["h2601", "*"]. 
    If ``key='*'``, all the data is aggregated on a single frame of borehole. 
    
    .. versionadded:: 0.2.3. 
       Add 08 new boreholes data from logging, strata, layer thicknesses and 
       rock_names. 
       
drop_observations: bool, default='False'
    Drop the ``remark`` column in the logging data if set to ``True``.  
    .. versionadded:: 0.1.5
    
Returns
---------
data : :class:`~watex.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} 
        The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
    target: {ndarray, Series} 
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame 
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
        .. versionadded:: 0.1.1
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
        .. versionadded:: 0.1.2
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.
    .. versionadded:: 0.1.2
X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of:
        
    .. math:: 
        
        \\text{shape}(X, y) =  1-  \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
        \\text{shape}(Xt, yt)= \\text{test_ratio} * (n_{samples}, n_{features}) *100
    
    where each row representing one sample and each column representing the 
    features. The second ndarray of shape(n_samples,) containing the target 
    samples.
     
Examples
--------
Let's say ,we do not have any idea of the columns that compose the target,
thus, the best approach is to run the function without passing any parameters::

>>> from watex.datasets.dload import load_hlogs 
>>> b= load_hlogs()
>>> b.target_names 
['aquifer_group',
 'pumping_level',
 'aquifer_thickness',
 'hole_depth',
 'pumping_depth',
 'section_aperture',
 'k',
 'kp',
 'r',
 'rp',
 'remark']
>>> # Let's say we are interested of the targets 'pumping_level' and 
>>> # 'aquifer_thickness' and returns `y' 
>>> _, y = load_hlogs (as_frame=True, # return as frame X and y
                       tnames =['pumping_level','aquifer_thickness'], 
                       )
>>> list(y.columns)
... ['pumping_level', 'aquifer_thickness']
 
"""
def load_nlogs (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None, 
    years=None, 
    split_X_y=False, 
    test_ratio=.3 , 
    tag=None, 
    tnames=None, 
    data_names=None, 
    samples=None, 
    seed =None, 
    shuffle =False, 
    **kws
    ): 

    drop_display_rate = kws.pop("drop_display_rate", True)
    key = key or 'b0' 
    # assertion error if key does not exist. 
    available_sets = {
        "b0", 
        "ns", 
        "hydrogeological", 
        "engineering", 
        "subsidence", 
        "ls"
        }
    is_keys = set ( list(available_sets))
    key = key_checker(key, is_keys, deep_search=True )
    key = "b0" if key in ("b0", "hydrogeological") else (
          "ns" if key in ("ns",  "engineering") else ( 
          "ls" if key in ("ls", "subsidence") else key )
        )
    assert key in (is_keys), (
        f"wrong key {key!r}. Expect {smart_format(is_keys, 'or')}")
    
    # read datafile
    if key in ("b0", "ns"):  
        data_file =f"nlogs{'+' if key=='ns' else ''}.csv" 
    else: data_file = "n.npz"
    with resources.path (DMODULE , data_file) as p : 
        data_file = str(p)
  
    if key=='ls': 
        # use tnames and years 
        # interchangeability 
        years = tnames or years 
        data , feature_names, target_columns= _get_subsidence_data(
            data_file, years = years or "2022",
            drop_display_rate= drop_display_rate )
        # reset tnames to fit the target columns
        tnames=target_columns 
    else: 
        data = pd.read_csv( data_file )
        # since target and columns are alread set 
        # for land subsidence data, then 
        # go for "ns" and "b0" to
        # set up features and target names
        feature_names = (list( data.columns [:21 ])  + [
            'filter_pipe_diameter']) if key=='b0' else list(
                filter(lambda item: item!='ground_height_distance',
                       data.columns)
                ) 
        target_columns = ['static_water_level', 'drawdown', 'water_inflow', 
                          'unit_water_inflow', 'water_inflow_in_m3_d'
                          ] if key=='b0' else  ['ground_height_distance']
    # cast values to numeric 
    data = to_numeric_dtypes( data) 
    samples = samples or "*"
    data = random_sampling(data, samples = samples, random_state= seed, 
                            shuffle= shuffle) 
    # reverify the tnames if given 
    # target columns 
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    try : 
        existfeatures(data[target_columns], tnames)
    except Exception as error: 
        # get valueError message and replace Feature by target
        verb ="s are" if len(target_columns) > 2 else " is"
        msg = (f" Valid target{verb}: {smart_format(target_columns, 'or')}")
        raise ValueError(str(error).replace(
            'Features'.replace('s', ''), 'Target(s)')+msg)
        
    # read dataframe and separate data to target. 
    frame, data, target = _to_dataframe(
        data, feature_names = feature_names, tnames = tnames, 
        target=data[tnames].values 
        )
    # for consistency, re-cast values to numeric 
    frame = to_numeric_dtypes(frame)
        
    if split_X_y: 
        X, Xt = split_train_test_by_id (
            data = frame , test_ratio= assert_ratio(test_ratio), 
            keep_colindex= False )
        
        y = X[tnames] 
        X.drop(columns =target_columns, inplace =True)
        yt = Xt[tnames]
        Xt.drop(columns =target_columns, inplace =True)
        
        return  (X, Xt, y, yt ) if as_frame else (
            X.values, Xt.values, y.values , yt.values )

    if return_X_y: 
        return (data.values, target.values) if not as_frame else (
            data, target) 
    
    # return frame if as_frame simply 
    if as_frame: 
        return frame 
    # upload the description file.
    descr_suffix= {"b0": '', "ns":"+", "ls":"++"}
    fdescr = description_loader(
        descr_module=DESCR,descr_file=f"nanshang{descr_suffix.get(key)}.rst")

    return Boxspace(
        data=data.values,
        target=target.values,
        frame=frame,
        tnames=tnames,
        target_names = target_columns,
        DESCR= fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
    )
 
load_nlogs.__doc__="""\
Load the Nanshang Engineering and hydrogeological drilling dataset.

Dataset contains multi-target and can be used for a classification or 
regression problem.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a Bowlspace object. See
    below for more information about the `data` and `target` object.

as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.

split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) with the according to the test size ratio. 
    
test_ratio: float, default is {{.3}} i.e. 30% (X, y)
    The ratio to split the data into training (X, y)  and testing (Xt, yt) set 
    respectively. 
    
tnames: str, optional 
    the name of the target to retreive. If ``None`` the full target columns 
    are collected and compose a multioutput `y`. For a singular classification 
    or regression problem, it is recommended to indicate the name of the target 
    that is needed for the learning task. When collecting data for land 
    subsidence with ``key="ls"``, `tnames` and `years` are used 
    interchangeability. 
    
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~watex.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
    
key: str, default='b0'
    Kind of drilling data to fetch. Can also be the borehole ["ns", "ls"]. The 
    ``ns`` data refer mostly to engineering drilling whereas the ``b0`` refers 
    to pure hydrogeological drillings. In the former case, the 
    ``'ground_height_distance'`` attribute used to control soil settlement is 
    the target while the latter targets fit the water inflow, the drawdown and 
    the static water level. The "ls" key is used for collection the times 
    series land subsidence data from 2015-2018. It should be used in combinaison
    with the `years` parameter for collecting the specific year data. The 
    default land-subsidence data is ``2022``. 
    
years: str, default="2022" 
   the year of land subsidence. Note that land subsidence data are collected 
   from 2015 to 2022. For instance to select two years subsidence, use 
   space between years like ``years ="2015 2022"``. The star ``*`` argument 
   can be used for selecting all years data. 
   
   .. versionadded:: 0.2.7 
      Years of Nanshan land subsidence data collected are added. Use key 
      `ls` and `years` for retrieving the subsidence data of each year. 
      
samples: int,optional 
   Ratio or number of items from axis to fetch in the data. fetch all data if 
   `samples` is ``None``.  
   
seed: int, array-like, BitGenerator, np.random.RandomState, \
    np.random.Generator, optional
   If int, array-like, or BitGenerator, seed for random number generator. 
   If np.random.RandomState or np.random.Generator, use as given.
   
shuffle: bool, default =False, 
   If ``True``, borehole data should be shuffling before sampling. 
   
drop_display_rate: bool, default=True 
  Display the rate is used for image visualization. To increase the image 
  pixels. 
  
  .. versionadded: 0.2.7 
    Drop rate for pixels increasing during visualization of land 
    subsidence. 

Returns
---------
data : :class:`~watex.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} 
        The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
    target: {ndarray, Series} 
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    frame: DataFrame 
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
        .. versionadded:: 0.1.1
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
        .. versionadded:: 0.1.2
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.
    .. versionadded:: 0.1.2
X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of:
        
    .. math:: 
        
        \\text{shape}(X, y) =  1-  \\text{test_ratio} * (n_{samples}, n_{features}) *100
        
        \\text{shape}(Xt, yt)= \\text{test_ratio} * (n_{samples}, n_{features}) *100
    
    where each row representing one sample and each column representing the 
    features. The second ndarray of shape(n_samples,) containing the target 
    samples.
     
Examples
--------
Let's say ,we do not have any idea of the columns that compose the target,
thus, the best approach is to run the function without passing any parameters 
and then `DESCR` attributes to get the unit of each attribute::

>>> from watex.datasets.dload import load_nlogs
>>> b= load_nlogs()
>>> b.target_names
Out[241]: 
['static_water_level',
 'drawdown',
 'water_inflow',
 'unit_water_inflow',
 'water_inflow_in_m3_d']
>>> b.DESCR
... (...)
>>> # Let's say we are interested of the targets 'drawdown' and 
>>> # 'static_water_level' and returns `y' 
>>> _, y = load_nlogs (as_frame=True, # return as frame X and y
                       tnames =['drawdown','static_water_level'], )
>>> list(y.columns)
... ['drawdown', 'static_water_level']
>>> y.head(2) 
   drawdown  static_water_level
0     70.03                4.21
1      7.38                3.60
>>> # let say we want subsidence data of 2015 and 2018 with the 
>>> # diplay resolution rate. Because the display is removed, we must set  
>>> # it to False so keep it included in the data. 
>>> n= load_nlogs (key ='ls', samples = 3 , years = "2015 2018 disp",
                   drop_display_rate =False )
>>> n.frame  
        easting      northing   longitude  ...      2015       2018  disp_rate
0  2.531191e+06  1.973515e+07  113.291328  ... -0.494959 -27.531837  -7.352538
1  2.531536e+06  1.973519e+07  113.291847  ... -1.104473 -21.852705  -7.999145
2  2.531479e+06  1.973520e+07  113.291847  ... -1.139404 -22.022655  -7.894940
"""
    
def load_mxs (
    *,  return_X_y=False, 
    as_frame =False, 
    key =None,  
    tag =None, 
    samples=None, 
    tnames = None , 
    data_names=None, 
    split_X_y=False, 
    seed = None, 
    shuffle=False,
    test_ratio=.2,  
    **kws): 
    import joblib

    drop_observations =kws.pop("drop_observations", False)
    
    target_map= { 
        0: '1',
        1: '11*', 
        2: '2', 
        3: '2*', 
        4: '3', 
        5: '33*'
        }
    
    add = {"data": ('data', ) , '*': (
        'X_train','X_test' , 'y_train','y_test' ), 
        }
    av= {"sparse": ('X_csr', 'ymxs_transf'), 
         "scale": ('Xsc', 'ymxs_transf'), 
         "train": ( 'X_train', 'y_train'), 
         "test": ('X_test', 'y_test'), 
         'numeric': ( 'Xnm', 'ymxs_transf'), 
         'raw': ('X', 'y')
         }
    
    if key is None: 
        key='data'
    data_file ='mxs.joblib'
    with resources.path (DMODULE , data_file) as p : 
        data_file = str(p)
        
    data_dict = joblib.load (data_file )
    # assertion error if key does not exist. 
    available_sets = set (list( av.keys() ) + list( add.keys()))

    msg = (f"key {key!r} does not exist yet, expect"
           f" {smart_format(available_sets, 'or')}")
    assert str(key).lower() in available_sets , msg
    # manage sampling 
    # by default output 50% data 
    samples= samples or .50 
    
    if split_X_y: 
        from ..exlib import train_test_split
        data = tuple ([data_dict [k] for k in add ['*'] ] )
        # concatenate the CSR matrix 
        X_csr = scipy.sparse.csc_matrix (np.concatenate (
            (data[0].toarray(), data[1].toarray()))) 
        y= np.concatenate ((data[-2], data[-1]))
        # resampling 
        data = (random_sampling(d, samples = samples,random_state= seed , 
                                shuffle= shuffle) for d in (X_csr, y ) 
                )
        # split now
        return train_test_split (*tuple ( data ),random_state = seed, 
                                 test_size =assert_ratio (test_ratio),
                                 shuffle = shuffle)
    # Append Xy to Boxspace if 
    # return_X_y is not set explicitly.
    Xy = dict() 
    # if for return X and y if k is not None 
    if key is not None and key !="data": 
        if key not in  av.keys():
            key ='raw'
        X, y =  tuple ( [ data_dict[k]  for k in av [key]] ) 

        X = random_sampling(X, samples = samples,random_state= seed , 
                            shuffle= shuffle)
        y = random_sampling(y, samples = samples, random_state= seed, 
                            shuffle= shuffle
                               )
        if return_X_y: 
            return (  X, y )  if as_frame or key =='sparse' else (
                np.array(X), np.array(y))
        
        # if return_X_y is not True 
        Xy ['X']=X ; Xy ['y']=y 
        # Initialize key to 'data' to 
        # append the remain data 
        key ='data'

    data = data_dict.get(key)  
    if drop_observations: 
        data.drop (columns = "remark", inplace = True )
        
    frame = None
    feature_names = list(data.columns [:13] ) 
    target_columns = list(data.columns [13:])
    
    tnames = tnames or target_columns
    # control the existence of the tnames to retreive
    data = random_sampling(data, samples = samples, random_state= seed, 
                           shuffle= shuffle)
    if as_frame:
        frame, data, target = _to_dataframe(
            data, feature_names = feature_names, tnames = tnames, 
            target=data[tnames].values 
            )
        frame = to_numeric_dtypes(frame)

    return Boxspace(
        data=data.values,
        target=data[tnames].values,
        frame=data,
        tnames=tnames,
        target_names = target_columns,
        target_map = target_map, 
        nga_labels = data_dict.get('nga_labels'), 
        #XXX Add description 
        DESCR= '', # fdescr,
        feature_names=feature_names,
        filename=data_file,
        data_module=DMODULE,
        **Xy
        )
    
load_mxs.__doc__="""\
Load the dataset after implementing the mixture learning strategy (MXS).

Dataset is composed of 11 boreholes merged with multiple-target that can be 
used for a classification problem.

Parameters
----------
return_X_y : bool, default=False
    If True, returns ``(data, target)`` instead of a Bowlspace object. See
    below for more information about the `data` and `target` object.
    
as_frame : bool, default=False
    If True, the data is a pandas DataFrame including columns with
    appropriate dtypes (numeric). The target is
    a pandas DataFrame or Series depending on the number of target columns.
    If `return_X_y` is True, then (`data`, `target`) will be pandas
    DataFrames or Series as described below.

split_X_y: bool, default=False,
    If True, the data is splitted to hold the training set (X, y)  and the 
    testing set (Xt, yt) based on to the `test_ratio` value.

tnames: str, optional 
    the name of the target to retrieve. If ``None`` the full target columns 
    are collected and compose a multioutput `y`. For a singular classification 
    or regression problem, it is recommended to indicate the name of the target 
    that is needed for the learning task. 
(tag, data_names): None
    `tag` and `data_names` do nothing. just for API purpose and to allow 
    fetching the same data uing the func:`~watex.data.fetch_data` since the 
    latter already holds `tag` and `data_names` as parameters. 
    
samples: int,optional 
   Ratio or number of items from axis to fetch in the data. 
   Default = .5 if `samples` is ``None``.

key: str, default='data'
    Kind of MXS data to fetch. Can also be: 
        
        - "sparse": for a compressed sparsed row matrix format of train set X. 
        - "scale": returns a scaled X using the standardization strategy 
        - "num": Exclusive numerical data and exclude the 'strata' feature.
        - "test": test data `X` and `y` 
        - "train": train data `X` and  `y` with preprocessing already performed
        - "raw": for original dataset X and y  with no preprocessing 
        - "data": Default when key is not supplied. It returns 
          the :class:`Bowlspace` objects.
        
    When k is not supplied, "data" is used instead and return a 
    :class:`Bowlspace` objects. where: 
        - target_map: is the mapping of MXS labels in the target y. 
        - nga_labels: is the y predicted for Naive Group of Aquifer. 

drop_observations: bool, default='False'
    Drop the ``remark`` column in the logging data if set to ``True``. 
    
seed: int, array-like, BitGenerator, np.random.RandomState, \
    np.random.Generator, optional
   If int, array-like, or BitGenerator, seed for random number generator. 
   If np.random.RandomState or np.random.Generator, use as given.
   
shuffle: bool, default =False, 
   If ``True``, borehole data should be shuffling before sampling. 
   
test_ratio: float, default is 0.2 i.e. 20% (X, y)
    The ratio to split the data into training (X, y) and testing (Xt, yt) set 
    respectively.
    
Returns
---------
data : :class:`~watex.utils.Boxspace`
    Dictionary-like object, with the following attributes.
    data : {ndarray, dataframe} 
        The data matrix. If ``as_frame=True``, `data` will be a pandas DataFrame.
    target: {ndarray, Series} 
        The classification target. If `as_frame=True`, `target` will be
        a pandas Series.
    feature_names: list
        The names of the dataset columns.
    target_names: list
        The names of target classes.
    target_map: dict, 
       is the mapping of MXS labels in the target y. 
    nga_labels: arryalike 1D, 
       is the y predicted for Naive Group of Aquifer. 
    frame: DataFrame 
        Only present when `as_frame=True`. DataFrame with `data` and
        `target`.
    DESCR: str
        The full description of the dataset.
    filename: str
        The path to the location of the data.
data, target: tuple if ``return_X_y`` is True
    A tuple of two ndarray. The first containing a 2D array of shape
    (n_samples, n_features) with each row representing one sample and
    each column representing the features. The second ndarray of shape
    (n_samples,) containing the target samples.

X, Xt, y, yt: Tuple if ``split_X_y`` is True 
    A tuple of two ndarray (X, Xt). The first containing a 2D array of
    training and test data whereas `y` and `yt` are training and test labels.
    The number of samples are based on the `test_ratio`. 
 
Examples
--------
>>> from watex.datasets.dload import load_mxs  
>>> load_mxs (return_X_y= True, key ='sparse', samples ='*')
(<1038x21 sparse matrix of type '<class 'numpy.float64'>'
 	with 8298 stored elements in Compressed Sparse Row format>,
 array([1, 1, 1, ..., 5, 5, 5], dtype=int64))
 
"""  
def _get_subsidence_data (
        data_file, /, 
        years: str="2022", 
        drop_display_rate: bool=... 
        ): 
    """Read, parse features and target for Nanshan land subsidence data
    
    Parameters 
    ------------
    data_file: str, Pathlike object 
       Full path to the object to read.
    years: str, default=2022 
        year of subsidence data collected. To collect the value of subsidence 
        of all years, set ``years="*"``
        
    drop_display_rate: bool, default=False, 
       Rate of display for visualisation in Goldern software. 
       
    Returns 
    --------
    data, feature_names, target_columns: pd.DataFrame, list
      DataFrame and list of features and targets. 
   
    """
    columns =['easting',
             'northing',
             'longitude',
             'latitude',
             '2015',
             '2016',
             '2017',
             '2018',
             '2019',
             '2020',
             '2021',
             '2022',
             'disp_rate'
             ]
    data = read_data ( data_file, columns = columns )
    if drop_display_rate: 
        data.pop('disp_rate')
        columns =columns [: -1]
        # remove display rate if exists while 
        # it is set to True
        if isinstance ( years, str): 
           years = years.replace ("disp", "").replace (
               "_", ' ').replace ("rate", "")
        elif hasattr ( years, '__iter__'): 
            # maybe list etc 
            years = [str(i) for i in years if not (str(i).find(
                "disp") >= 0 or str(i).find("rate")>= 0) ] 

    if years !="*": 
        years = key_checker (years, valid_keys= data.columns ,
                     pattern =r'[#&*@!,;\s-]\s*', deep_search=True)
        years = [years] if isinstance ( years, str) else years 
    else : 
        years = columns[4:] # target columns 
    # recheck duplicates and remove it  
    years = sorted(set ((str(y) for y in years)))
    feature_names = columns[: 4 ]
    target_columns = years 
    data = data [ feature_names + target_columns ]
    
    return  data,  feature_names, target_columns     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
