# -*- coding: utf-8 -*-
# Licence:BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

from __future__ import annotations 
import os 
import sys 
import logging 
import random
import warnings 
 
# set the package name for consistency checker 
sys.path.insert(0, os.path.dirname(__file__))  
for p in ('.','..' ,'./hlearn'): 
    sys.path.insert(0,  os.path.abspath(p)) 
    
# assert package 
if  __package__ is None: 
    sys.path.append( os.path.dirname(__file__))
    __package__ ='hlearn'

# configure the logger file
# from ._hlearnlog import hlearnlog
try: 
    conffile = os.path.join(
        os.path.dirname(__file__),  "hlearn/hlog.yml")
    if not os.path.isfile (conffile ): 
        raise 
except: 
    conffile = os.path.join(
        os.path.dirname(__file__), "hlog.yml")

# generated version by setuptools_scm 
__version__ = '0.1.0' 

# # set loging Level
logging.getLogger(__name__)#.setLevel(logging.WARNING)
# disable the matplotlib font manager logger.
logging.getLogger('matplotlib.font_manager').disabled = True
# or ust suppress the DEBUG messages but not the others from that logger.
# logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# setting up
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
try:
    # This variable is injected in the __builtins__ by the build process. 
    __HLEARN_SETUP__  # type: ignore
except NameError:
    __HLEARN_SETUP__  = False

if __HLEARN_SETUP__ :
    sys.stderr.write("Partial import of hlearn during the build process.\n")
else:
    from . import _distributor_init  # noqa: F401
    from . import _build  # noqa: F401
    from .utils._show_versions import show_versions
    
#https://github.com/pandas-dev/pandas
# Let users know if they're missing any of our hard dependencies
_main_dependencies = ("numpy", "scipy", "sklearn", "matplotlib", 
                      "pandas","seaborn")
_missing_dependencies = []

for _dependency in _main_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(
            f"{'scikit-learn' if _dependency=='sklearn' else _dependency }: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _main_dependencies, _dependency, _missing_dependencies

# Try to suppress pandas future warnings
# and reduce verbosity.
# Setup hlearn public API  
with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=UserWarning)
    import hlearn.externals as sklearn 

from .datasets import ( 
    fetch_data, 
    ) 
from .methods import ( 
    Structural, 
    Structures, 
    MXS, 
    )

from .view import ( 
    EvalPlot, 
    plotLearningInspections, 
    plotSilhouette,
    plotDendrogram, 
    plotProjection, 
    )

from .utils import ( 
    read_data,
    cleaner, 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier,
    select_base_stratum , 
    reduce_samples , 
    make_MXS_labels, 
    predict_NGA_labels, 
    classify_k,  
    plot_elbow, 
    plot_clusters, 
    plot_pca_components, 
    plot_naive_dendrogram, 
    plot_learning_curves, 
    plot_confusion_matrices, 
    plot_sbs_feature_selection, 
    plot_regularization_path, 
    plot_rf_feature_importances, 
    plot_logging, 
    plot_silhouette, 
    plot_profiling,
    plot_confidence_in,
    )

try : 
    from .utils import ( 
        selectfeatures, 
        naive_imputer, 
        naive_scaler,  
        make_naive_pipe, 
        bi_selector, 
        )
except ImportError :
    pass 

def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("hlearn_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
   
__doc__= """\
hydro-learn: An intelligent solver for hydrogeology engineering issues
=======================================================================

Hydro-learn is a Python-based package for solving hydro-geology engineering 
issues. From methodologies based on Machine Learning,It brings novel 
approaches for reducing numerous losses during the hydrogeological
exploration projects. It allows to:

- reduce the cost of permeability coefficient (k) data collection during the 
  engineering projects,
- guide drillers for to locating the drilling operations,
- predict the water content in the well such as the level of water inrush, ...

.. _hlearn: https://github.com/WEgeophysics/hydro-learn/

"""
#  __all__ is used to display a few public API. 
# the public API is determined
# based on the documentation.
    
__all__ = [ 
    "sklearn", 
    "fetch_data",
    "Structural", 
    "Structures", 
    "MXS", 
    "EvalPlot", 
    "plotLearningInspections", 
    "plotSilhouette",
    "plotDendrogram", 
    "plotProjection", 
    "plotAnomaly", 
    "vesSelector", 
    "erpSelector", 
    "read_data",
    "erpSmartDetector", 
    "plot_confidence_in", 
    "reshape", 
    "to_numeric_dtypes", 
    "smart_label_classifier",
    "select_base_stratum" , 
    "reduce_samples" , 
    "make_MXS_labels", 
    "predict_NGA_labels", 
    "classify_k",  
    "plot_elbow", 
    "plot_clusters", 
    "plot_pca_components", 
    "plot_naive_dendrogram", 
    "plot_learning_curves", 
    "plot_confusion_matrices",  
    "plot_sbs_feature_selection", 
    "plot_regularization_path", 
    "plot_rf_feature_importances", 
    "plot_logging", 
    "plot_silhouette", 
    "plot_profiling", 
    "selectfeatures", 
    "naive_imputer", 
    "naive_scaler",  
    "make_naive_pipe", 
    "bi_selector", 
    "show_versions",
    "cleaner", 
    ]

