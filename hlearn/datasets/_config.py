# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" 
Set all dataset.  
"""
from warnings import warn 

from ..utils.funcutils import ( 
    smart_format 
    )
from ..exceptions import DatasetError
from .._hlearnlog import hlearnlog

_logger = hlearnlog().get_hlearn_logger(__name__)

_DTAGS=(
    "hlogs",
    "nlogs", 
    "mxs", 
    )

from .dload import (
    load_hlogs,
    load_nlogs, 
    load_mxs, 
    ) 

__all__=[ 
         
         "load_hlogs",
         "load_nlogs", 
         "fetch_data",
         "load_mxs", 
         
         ]

def fetch_data (tag, **kws): 
    tag = _parse_tags(tag, multi_kind_dataset='nanshan')
    funcs= ( load_hlogs, load_nlogs, load_mxs ) 
    funcns = list (map(lambda f: f.__name__.replace('load_', ''), funcs))
    if tag in (funcns): 
        func = funcs[funcns.index (tag)] 
    else : raise DatasetError( 
        f"Unknown data set {tag!r}. Expect {smart_format( funcns)}")
    
    return func (tag=tag, data_names=funcns, **kws) if callable (func) else None 


fetch_data.__doc__ ="""\
Fetch dataset from `tag`. 

A tag corresponds to the name area of data collection or each 
level of data processing. 

Parameters 
------------
tag: str, ['nlogs', 'hlogs', 'mxs', ]
    name of the area of data to fetch. 

Returns
-------
dict, X, y : frame of :class:`~hlearn.utils.box.Boxspace` object 
   
"""    

def _parse_tags (tag, multi_kind_dataset ='nanshan'): 
    """ Parse and sanitize tag to match the different type of datasets.
    
    In principle, only the 'Bagoue' datasets is allowed to contain a tag 
    composed of two words i.e. 'Bagoue' + '<kind_of_data>'. For instance 
    ``bagoue pipe`` fetchs only the pipeline used for Bagoue case study  
    data preprocessing and so on. 
    However , for other type of dataset, it a second word <kind_of_data> is 
    passed, it should merely discarded. 
    """ 
    tag = str(tag);  t = tag.strip().split() 
    
    if len(t) ==1 : 
        if t[0].lower() not in _DTAGS: 
            tag = multi_kind_dataset +' ' + t[0]
            
            warn(f"Fetching {multi_kind_dataset.title()!r} data without"
                 " explicitly prefixing the kind of data with the area"
                 " name will raise an error. In future, the argument"
                f" should be '{tag}' instead.", FutureWarning 
                 )
    elif len(t) >1 : 
        # only the multi kind dataset is allowed 
        # to contain two words for fetching data 
        if t[0].lower() !=multi_kind_dataset: 
            tag = t[0].lower() # skip the second word 
    return tag 

from ..utils.funcutils import listing_items_format

_l=[ "{:<7}: {:<7}()".format(s.upper() , 'load_'+s ) for s in _DTAGS ] 
_LST = listing_items_format(
    _l, 
    "Fetch data using 'load_<type_of_data|area_name>'like", 
    " or using ufunc 'fetch_data (<type_of_data|area_name>)'.",
    inline=True , verbose= False, 
)

