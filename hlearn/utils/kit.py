# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 22:47:18 2023

@author: Daniel
"""
import inspect 
import warnings
import numpy as np 
# import pandas as pd 
 
from .._hlearnlog import hlearnlog
from ..externals.sklean import ( 
    BaseEstimator, TransformerMixin, 
    StandardScaler, MinMaxScaler, 
    OrdinalEncoder, OneHotEncoder
    )
_logger = hlearnlog().get_hlearn_logger(__name__)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Select data from specific attributes for column transformer. 
    
    Select only numerical or categorial columns for operations. Work as the
    same like sckit-learn `make_colum_tranformer` 
    
    Arguments  
    ----------
    *attribute_names*: list or array_like 
        List of  the main columns to keep the data 
        
    *select_type*: str 
        Automatic numerical and categorial selector. If `select_type` is 
        ``num``, only numerical values in dataframe are retrieved else 
        ``cat`` for categorials attributes.
            
    Returns
    -------
    X: ndarray 
        New array with composed of data of selected `attribute_names`.
            
    Examples 
    ---------
    >>> from hlearn.transformers import DataFrameSelector 
    >>> from hlearn.utils.mlutils import load_data   
    >>> df = mlfunc.load_data('data/geo_fdata')
    >>> XObj = DataFrameSelector(attribute_names=['power','magnitude','sfi'],
    ...                          select_type=None)
    >>> cdf = XObj.fit_transform(df)
    
    """  
    def __init__(self, attribute_names=None, select_type =None): 
        self._logging= hlearnlog().get_hlearn_logger(self.__class__.__name__)
        self.attribute_names = attribute_names 
        self.select_type = select_type 
        
    def fit(self, X, y=None): 
        """ 
        Select the Data frame 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `DataFrameSelector` instance 
            returns ``self`` for easy method chaining.
        
        """
        return self
    
    def transform(self, X): 
        """ Transform data and return numerical or categorial values.
        
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        """
       
        if isinstance(self.attribute_names, str): 
            self.attribute_names =[self.attribute_names]
            
        if self.attribute_names is not None: 
            t_= []
            for in_attr in self.attribute_names: 
                for attr_ in X.columns: 
                    if in_attr.lower()== attr_.lower(): 
                        t_.append(attr_)
                        break 
                    
            if len(t_)==0: 
                self._logging.warn(f' `{self.attribute_names}` not found in the'
                                   '`{X.columns}`.')
                warnings.warn('None attribute in the dataframe match'
                              f'`{self.attribute_names}.')
                
            if len(t_) != len(self.attribute_names): 
                mm_= set(self.attribute_names).difference(set(t_))
                warnings.warn(
                    f'Value{"s" if len(mm_)>1 else""} {list(mm_)} not found.'
                    f" Only `{t_}`match{'es' if len(t_) <1 else ''}"
                    " the dataframe features.")
                self._logging.warning(
                    f'Only `{t_}` can be considered as dataframe attributes.')
                                   
            self.attribute_names =t_
            
            return X[self.attribute_names].values 
        
        try: 
            if self.select_type.lower().find('num')>=0:
                self.select_type =='num'
            elif self.select_type.lower().find('cat')>=0: 
                self.select_type =='cat'
            else: self.select_type =None 
            
        except:
            warnings.warn(f'`Select_type`` given argument ``{self.select_type}``'
                         ' seems to be wrong. Should defaultly return the '
                         'Dataframe value.', RuntimeWarning)
            self._logging.warnings('A given argument `select_type`seems to be'
                                   'wrong %s. Use ``cat`` or ``num`` for '
                                   'categorical or numerical attributes '
                                   'respectively.'% inspect.signature(self.__init__))
            self.select_type =None 
        
        if self.select_type is None:
            warnings.warn('Arguments of `%s` arguments %s are all None. Should'
                          ' returns the dataframe values.'% (repr(self),
                              inspect.signature (self.__init__)))
            
            self._logging.warning('Object arguments are None.'
                               ' Should return the dataframe values.')
            return X.values 
        
        if self.select_type =='num':
            obj_columns= X.select_dtypes(include='number').columns.tolist()

        elif self.select_type =='cat': 
            obj_columns= X.select_dtypes(include=['object']).columns.tolist() 
 
        self.attribute_names = obj_columns 
        
        return X[self.attribute_names].values 
        
    def __repr__(self):
        return self.__class__.__name__  
    
    
class FrameUnion (BaseEstimator, TransformerMixin) : 
    """ Unified categorial and numerical features after scaling and 
    and categorial features encoded.
    
    Use :class:`~hlearn.tranformers.DataframeSelector` class to define 
    the categorial features and numerical features.
    
    Arguments
    ---------
    num_attributes: list 
        List of numerical attributes 
        
    cat_attributes: list 
        list of categorial attributes 
        
    scale: bool 
        Features scaling. Default is ``True`` and use 
        `:class:~sklearn.preprocessing.StandarScaler` 
        
    imput_data: bool , 
        Replace the missing data. Default is ``True`` and use 
        :attr:`~sklearn.impute.SimpleImputer.strategy`. 
        
    param_search: bool, 
        If `num_attributes` and `cat_attributes`are None, the numerical 
        features and categorial features` should be found automatically.
        Default is ``True``
        
    scale_mode:bool, 
        Mode of data scaling. Default is ``StandardScaler``but can be 
        a ``MinMaxScaler`` 
        
    encode_mode: bool, 
        Mode of data encoding. Default is ``OrdinalEncoder`` but can be 
        ``OneHotEncoder`` but creating a sparse matrix. Once selected, 
        the new shape of ``X`` should be different from the original 
        shape. 
    
    """  
    def __init__(
        self,
        num_attributes =None , 
        cat_attributes =None,
        scale =True,
        imput_data=True,
        encode =True, 
        param_search ='auto', 
        strategy ='median', 
        scale_mode ='StandardScaler', 
        encode_mode ='OrdinalEncoder'
        ): 
        
        self._logging = hlearnlog().get_hlearn_logger(self.__class__.__name__)
        
        self.num_attributes = num_attributes 
        self.cat_attributes = cat_attributes 
        self.param_search = param_search 
        self.imput_data = imput_data 
        self.strategy =strategy 
        self.scale = scale
        self.encode = encode 
        self.scale_mode = scale_mode
        self.encode_mode = encode_mode
        
        self.X_=None 
        self.X_num_= None 
        self.X_cat_ =None
        self.num_attributes_=None
        self.cat_attributes_=None 
        self.attributes_=None 
        
    def fit(self, X, y=None): 
        """
        Does nothing. Just for scikit-learn purpose. 
        """
        return self
    
    def transform(self, X): 
        """ Transform data and return X numerical and categorial encoded 
        values.
        
        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            
        Returns 
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            transformed arraylike, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            
        """
        
        if self.scale_mode.lower().find('stand')>=0: 
            self.scale_mode = 'StandardScaler'
        elif self.scale_mode.lower().find('min')>=0: 
            self.scale_mode = 'MinMaxScaler'
        if self.encode_mode.lower().find('ordinal')>=0: 
            self.encode_mode = 'OrdinalEncoder'
            
        elif self.encode_mode.lower().find('hot') >=0: 
            self.encode_mode = 'OneHotEncoder'
            
        numObj = DataFrameSelector(attribute_names= self.num_attributes, 
                                         select_type='num')
        catObj =DataFrameSelector(attribute_names= self.cat_attributes, 
                                         select_type='cat')
        num_arrayObj = numObj.fit_transform(X)
        cat_arrayObj = catObj.fit_transform(X)
        self.num_attributes_ = numObj.attribute_names 
        self.cat_attributes_ = catObj.attribute_names 
        
        self.attributes_ = self.num_attributes_ + self.cat_attributes_ 
        
        self.X_num_= num_arrayObj.copy()
        self.X_cat_ =cat_arrayObj.copy()
        self.X_ = np.c_[self.X_num_, self.X_cat_]
        
        if self.imput_data : 
            from sklearn.impute import SimpleImputer
            imputer_obj = SimpleImputer(missing_values=np.nan, 
                                        strategy=self.strategy)
            num_arrayObj =imputer_obj.fit_transform(num_arrayObj)
            
        if self.scale :
            if self.scale_mode == 'StandardScaler': 
                scaler = StandardScaler()
            if self.scale_mode =='MinMaxScaler':
                scaler = MinMaxScaler()
        
            num_arrayObj = scaler.fit_transform(num_arrayObj)
            
        if self.encode : 
            if self.encode_mode =='OrdinalEncoder': 
                encoder = OrdinalEncoder()
            elif self.encode_mode =='OneHotEncoder':
                encoder = OneHotEncoder(sparse_output=True)
            cat_arrayObj= encoder.fit_transform(cat_arrayObj )
            # sparse matrix of type class <'numpy.float64'>' stored 
            # element in compressed sparses raw format . To convert the sense 
            # matrix to numpy array , we need to just call 'to_array()'.
            warnings.warn(f'Sparse matrix `{cat_arrayObj.shape!r}` is converted'
                          ' in dense Numpy array.', UserWarning)
            # cat_arrayObj= cat_arrayObj.toarray()

        try: 
            X= np.c_[num_arrayObj,cat_arrayObj]
            
        except ValueError: 
            # For consistency use the np.concatenate rather than np.c_
            X= np.concatenate((num_arrayObj,cat_arrayObj), axis =1)
        
        if self.encode_mode =='OneHotEncoder':
            warnings.warn('Use `OneHotEncoder` to encode categorial features'
                          ' generates a Sparse matrix. X is henceforth '
                          ' composed of sparse matrix. The new dimension is'
                          ' {0} rather than {1}.'.format(X.shape,
                             self.X_.shape), UserWarning)
            self._logging.info('X become a spared matrix. The new shape is'
                               '{X.shape!r} against the orignal '
                               '{self.X_shape!r}')
            
        return X