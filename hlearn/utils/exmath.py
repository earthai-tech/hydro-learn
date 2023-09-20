# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Utilities to process and compute parameters. Module for Algebra calculus.
"""
from __future__ import annotations 
import copy 
import inspect 
import warnings 
import cmath 
from math import factorial, radians

import numpy as np
import pandas as pd 
from scipy.signal import argrelextrema 
from scipy.optimize import curve_fit
from scipy.integrate import quad 
from scipy.cluster.hierarchy import  linkage 
from scipy.linalg import lstsq
from scipy._lib._util import float_factorial
from scipy.spatial.distance import ( 
    pdist, squareform 
    )
import  matplotlib.pyplot as plt
from .._hlearnlog import hlearnlog
from .._docstring import refglossary
from ..decorators import ( 
    deprecated, 
    refAppender, 
    docSanitizer
)
from ..exceptions import ( 
    StationError, 
    VESError, 
    ERPError,
    EMError, 
    )
# from ..property import P
from .._typing import (
    T, 
    F,
    List, 
    Tuple,
    Union,
    ArrayLike,
    NDArray,
    DType,
    Optional,
    Sub, 
    SP, 
    Series, 
    DataFrame,
    EDIO,
    ZO
)
from .box import Boxspace 
from .funcutils import (
    _assert_all_types, 
    _validate_name_in, 
    assert_ratio,
    concat_array_from_list, 
    get_confidence_ratio,
    remove_outliers, 
    ellipsis2false,    
    smart_format,
    is_iterable, 
    reshape,
    ismissing,
    fillNaN, 
    spi,       
)
from .validator import ( 
    _is_arraylike_1d, 
    _validate_ves_operator, 
    _assert_z_or_edi_objs, 
    _validate_tensor,
    _is_numeric_dtype,
    check_consistency_size,
    check_y,
    check_array,
    assert_xy_in
    )

try: import scipy.stats as spstats
except: pass 

_logger =hlearnlog.get_hlearn_logger(__name__)

mu0 = 4 * np.pi * 1e-7 

def get_azimuth (
    xlon: str | ArrayLike, 
    ylat: str| ArrayLike, 
    *, 
    data: DataFrame =None, 
    utm_zone:str=None, 
    projection:str='ll', 
    isdeg:bool=True, 
    mode:str='soft', 
    extrapolate:bool =...,
    view:bool=..., 
    ): 
    """Compute azimuth from coordinate locations ( latitude,  longitude). 
    
    If `easting` and `northing` are given rather than `longitude` and  
    `latitude`, the projection should explicitely set to ``UTM`` to perform 
    the ideal conversion. However if mode is set to `soft` (default), the type
    of projection is automatically detected . Note that when UTM coordinates 
    are provided, `xlon` and `ylat` fit ``easting`` and ``northing`` 
    respectively.
    
    Parameters
    -----------
    xlon, ylat : Arraylike 1d or str, str 
       ArrayLike of easting/longitude and arraylike of nothing/latitude. They 
       should be one dimensional. In principle if data is supplied, they must 
       be series.  If `xlon` and `ylat` are given as string values, the 
       `data` must be supplied. xlon and ylat names must be included in the  
       dataframe otherwise an error raises. 
       
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       when `mode='soft'`,  the auto-detection will be triggered and find the 
       suitable coordinate system. However, it is recommended to explicitly 
       provide projection when data is in UTM coordinates. 
       Note that if `x` and `y` are composed of value greater than 180 degrees 
       for longitude and 90 degrees for latitude, and method is still in 
       the ``soft` mode, it should be considered as  longitude-latitude ``UTM``
       coordinates system. 
       
    isdeg: bool, default=True 
      By default xlon and xlat are in degree coordinates. If both arguments 
      are given in radians, set to ``False`` instead. 
      
    mode: str , ['soft'|'strict']
      ``strict`` mode does not convert any coordinates system to other at least
      it is explicitly set to `projection` whereas the `soft` does.
      
    extrapolate: bool, default=False 
      In principle, the azimuth is compute between two points. Thus, the number
      of values computed for :math:`N` stations should  be  :math:`N-1`. To fit
      values to match the number of size of the array, `extrapolate` should be 
      ``True``. In that case, the first station holds a <<fake>> azimuth as 
      the closer value computed from interpolation of all azimuths. 
      
    view: bool, default=False, 
       Quick view of the azimuth. It is usefull especially when 
       extrapolate is set to ``True``. 
       
    Return 
    --------
    azim: ArrayLike 
       Azimuth computed from locations. 
       
    Examples 
    ----------
    >>> import hlearn as wx 
    >>> from hlearn.utils.exmath import get_azimuth 
    >>> # generate a data from ERP 
    >>> data = wx.make_erp (n_stations =7 ).frame 
    >>> get_azimuth ( data.longitude, data.latitude)
    array([54.575, 54.575, 54.575, 54.575, 54.575, 54.575])
    >>> get_azimuth ( data.longitude, data.latitude, view =True, extrapolate=True)
    array([54.57500007, 54.575     , 54.575     , 54.575     , 54.575     ,
           54.575     , 54.575     ])
    
    """
    from ..site import Location 
    
    mode = str(mode).lower() 
    projection= str(projection).lower()
    extrapolate, view = ellipsis2false (extrapolate, view)

    xlon , ylat = assert_xy_in(xlon , ylat , data = data )
    
    if ( 
            xlon.max() > 180.  and ylat.max() > 90.  
            and projection=='ll' 
            and mode=='soft'
            ): 
        warnings.warn("xlon and ylat arguments are greater than 180 degrees."
                     " we assume the coordinates are UTM. Set explicitly"
                     " projection to ``UTM`` to avoid this warning.")
        projection='utm'
        
    if projection=='utm':
        if utm_zone is None: 
            raise TypeError ("utm_zone cannot be None when projection is UTM.")
            
        ylat , xlon = Location.to_latlon_in(
            xlon, ylat, utm_zone= utm_zone)
        
    if len(xlon) ==1 or len(ylat)==1: 
        msg = "Azimuth computation expects at least two points. Got 1"
        if mode=='soft': 
            warnings.warn(msg) 
            return 0. 
        
        raise TypeError(msg )
    # convert to radian 
    if isdeg: 
        xlon = np.deg2rad (xlon ) ; ylat = np.deg2rad ( ylat)
    
    dx = map (lambda ii: np.cos ( ylat[ii]) * np.sin( ylat [ii+1 ]) - 
        np.sin(ylat[ii]) * np.cos( ylat[ii+1]) * np.cos (xlon[ii+1]- xlon[ii]), 
        range (len(xlon)-1)
        )
    dy = map( lambda ii: np.cos (ylat[ii+1])* np.sin( xlon[ii+1]- xlon[ii]), 
                   range ( len(xlon)-1)
                   )
    # to deg 
    z = np.around ( np.rad2deg ( np.arctan2(list(dx) , list(dy) ) ), 3)  
    azim = z.copy() 
    if extrapolate: 
        # use mean azimum of the total area zone and 
        # recompute the position by interpolation 
        azim = np.hstack ( ( [z.mean(), z ]))
        # reset the interpolare value at the first position
        with warnings.catch_warnings():
            #warnings.filterwarnings(action='ignore', category=OptimizeWarning)
            warnings.simplefilter("ignore")
            azim [0] = scalePosition(azim )[0][0] 
        
    if view: 
        x = np.arange ( len(azim )) 
        fig,  ax = plt.subplots (1, 1, figsize = (10, 4))
        # add Nan to the first position of z 
        z = np.hstack (([np.nan], z )) if extrapolate else z 
       
        ax.plot (x, 
                 azim, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='extra-azimuth'
                 ) 
        
        ax.plot (x, 
                z, 
                'ok-', 
                label ='raw azimuth'
                )
        ax.legend ( ) 
        ax.set_xlabel ('x')
        ax.set_ylabel ('y') 

    return azim

def linkage_matrix(
    df: DataFrame ,
    columns:List[str] =None,  
    kind:str ='design', 
    metric:str ='euclidean',   
    method:str ='complete', 
    as_frame =False,
    optimal_ordering=False, 
 )->NDArray: 
    r""" Compute the distance matrix from the hierachical clustering algorithm
    
    Parameters 
    ------------ 
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions.
        
    optimal_ordering : bool, optional
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal. This results in a more intuitive
        tree structure when the data are visualized. defaults to False, because
        this algorithm can be slow, particularly on large datasets. See
        also :func:`scipy.cluster.hierarchy.linkage`. 
        
        
    Returns 
    --------
    row_clusters: linkage matrix 
        consist of several rows where each rw represents one merge. The first 
        and second columns denotes the most dissimilar members of each cluster 
        and the third columns reports the distance between those members 
        
        
    Linkage Methods 
    -----------------
    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

    * method='single' assigns

      .. math::
         d(u,v) = \min(dist(u[i],v[j]))

      for all points :math:`i` in cluster :math:`u` and
      :math:`j` in cluster :math:`v`. This is also known as the
      Nearest Point Algorithm.

    * method='complete' assigns

      .. math::
         d(u, v) = \max(dist(u[i],v[j]))

      for all points :math:`i` in cluster u and :math:`j` in
      cluster :math:`v`. This is also known by the Farthest Point
      Algorithm or Voor Hees Algorithm.

    * method='average' assigns

      .. math::
         d(u,v) = \sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

      for all points :math:`i` and :math:`j` where :math:`|u|`
      and :math:`|v|` are the cardinalities of clusters :math:`u`
      and :math:`v`, respectively. This is also called the UPGMA
      algorithm.

    * method='weighted' assigns

      .. math::
         d(u,v) = (dist(s,v) + dist(t,v))/2

      where cluster u was formed with cluster s and t and v
      is a remaining cluster in the forest (also called WPGMA).

    * method='centroid' assigns

      .. math::
         dist(s,t) = ||c_s-c_t||_2

      where :math:`c_s` and :math:`c_t` are the centroids of
      clusters :math:`s` and :math:`t`, respectively. When two
      clusters :math:`s` and :math:`t` are combined into a new
      cluster :math:`u`, the new centroid is computed over all the
      original objects in clusters :math:`s` and :math:`t`. The
      distance then becomes the Euclidean distance between the
      centroid of :math:`u` and the centroid of a remaining cluster
      :math:`v` in the forest. This is also known as the UPGMC
      algorithm.

    * method='median' assigns :math:`d(s,t)` like the ``centroid``
      method. When two clusters :math:`s` and :math:`t` are combined
      into a new cluster :math:`u`, the average of centroids s and t
      give the new centroid :math:`u`. This is also known as the
      WPGMC algorithm.

    * method='ward' uses the Ward variance minimization algorithm.
      The new entry :math:`d(u,v)` is computed as follows,

      .. math::

         d(u,v) = \sqrt{\frac{|v|+|s|}{T}d(v,s)^2 \\
                      + \frac{|v|+|t|}{T}d(v,t)^2 \\
                      - \frac{|v|}{T}d(s,t)^2}

      where :math:`u` is the newly joined cluster consisting of
      clusters :math:`s` and :math:`t`, :math:`v` is an unused
      cluster in the forest, :math:`T=|v|+|s|+|t|`, and
      :math:`|*|` is the cardinality of its argument. This is also
      known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    implementation may choose a different minimum than the MATLAB
    version.
    
    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
           algorithms", :arXiv:`1109.2378v1`.
    .. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal
           leaf ordering for hierarchical clustering", 2001. Bioinformatics
           :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    """
    df = _assert_all_types(df, pd.DataFrame, np.ndarray)
    
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("Number of columns must fit the shape of X."
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df.values if hasattr(df, 'columns') else df ,
                          columns = columns )
        
    kind= str(kind).lower().strip() 
    if kind not in ('squareform', 'condense', 'design'): 
        raise ValueError(f"Unknown method {method!r}. Expect 'squareform',"
                         " 'condense' or 'design'.")
        
    labels = [f'ID_{i}' for i in range(len(df))]
    if kind =='squareform': 
        row_dist = pd.DataFrame (squareform ( 
        pdist(df, metric= metric )), columns = labels  , 
        index = labels)
        row_clusters = linkage (row_dist, method =method, metric =metric
                                )
    if kind =='condens': 
        row_clusters = linkage (pdist(df, metric =metric), method =method
                                )
    if kind =='design': 
        row_clusters = linkage(df.values if hasattr (df, 'columns') else df, 
                               method = method, 
                               optimal_ordering=optimal_ordering )
        
    if as_frame: 
        row_clusters = pd.DataFrame ( row_clusters, 
                                     columns = [ 'row label 1', 
                                                'row lable 2', 
                                                'distance', 
                                                'no. of items in clust.'
                                                ], 
                                     index = ['cluster %d' % (i +1) for i in 
                                              range(row_clusters.shape[0])
                                              ]
                                     )
    return row_clusters 

def d_hanning_window(
        x: ArrayLike[DType[float]],
        xk: float , 
        W: int 
        )-> F: 
    """ Discrete hanning function.
    
    For futher details, please refer to https://doi.org/10.1190/1.2400625
    
    :param x: variable point along the window width
    :param xk: Center of the window `W`. It presumes to host the most weigth.   
    :param W: int, window-size; preferably set to odd number. It must be less than
          the dipole length. 
    :return: Anonymous function (x,xk, W) value 
    """
    # x =check_y (x, input_name ='x') 
    return  1/W * (1 + np.cos (
        2 * np.pi * (x-xk) /W)) if np.abs(x-xk) <= W/2 else  0.
    
def betaj (
        xj: int ,
        L: int , 
        W: int , 
        **kws
 )-> float : 
    """ Weight factor function for convoluting at station/site j.
    
    The function deals with the discrete hanning window based on ideas presented 
    in Torres-Verdin and Bostick, 1992, https://doi.org/10.1190/1.2400625.
    
    :param xj: int, position of the point to compute its weight. 
    :param W: int, window size, presumes to be the number of dipole. 
    :param L: int : length of dipole in meters 
    :param kws: dict , additional :func:`scipy.intergate.quad` functions.
    
    :return: Weight value at the position `xj`, prefix-`x`is used to specify  
        the direction. Commonly the survey direction is considered as `x`.
        
    :example: 
        >>> from hlearn.exmath import betaj 
        >>> # compute the weight point for window-size = 5 at position j =2
        >>> L= 1 ; W=5 
        >>> betaj (xj = 2 , L=L, W=W )
        ... 0.35136534572813144
    """
    if W < L : 
        raise ValueError("Window-size must be greater than the dipole length.")
        
    xk = W/2 
    # vec_betaj = np.vectorize( betaj ) ; vec_betaj(0, 1, 5)
    return  quad (d_hanning_window, xj - L/2 , xj +L/2, args=( xk, W), 
                  **kws)[0]

def rhoa2z ( 
        rhoa: NDArray[DType[T]], 
        phs:ArrayLike, 
        freq: ArrayLike
)-> NDArray[DType[T]]:
    r""" Convert apparent resistivity to impendance tensor z 
    
    :param rhoa: Apparent resistivity in :math:`\Omega.m` 
    :type rhoa: ndarray, shape (N, M) 
    
    :param phs: Phase in degrees 
    :type phs: ndarray, shape (N, M) 
    :param freq: Frequency in Hertz
    :type freq: array-like , shape (N, )
    :
    :return: Impendance tensor; Tensor is a complex number in :math:`\Omega`.  
    :rtype: ndarray, shape (N, M), dtype = 'complex' 
    
    :example: 
    >>> import numpy as np 
    >>> rhoa = np.array([1623.73691735])
    >>> phz = np.array([45.])
    >>> f = np.array ([1014])
    >>> rhoa2z(rhoa, phz, f)
    ... array([[2.54950976+2.54950976j]])
    
    """
    
    rhoa = np.array(rhoa); freq = np.array(freq) ; phs = np.array(phs) 
    
    if len(phs) != len(rhoa): 
        raise ValueError ("Phase and rhoa must have the same length."
                          f" {len(phs)} & {len(rhoa)} are given.")

    if len(freq) != len(rhoa): 
        raise ValueError("frequency and rhoa must have the same length."
                         "{len(freq} & {len(rhoa)} are given.")
        
    omega0 = 2 * np.pi * freq[:, None]
    z= np.sqrt(rhoa * omega0 * mu0 ) * (np.cos (
        np.deg2rad(phs)) + 1j * np.sin(np.deg2rad(phs)))
    
    return z 

def rhophi2z(rho, phi, freq):
    """
    Convert impedance-style information given in Rho/Phi format 
    into complex valued Z.

    Parameters 
    -----------
    rho: ArrayLike 1D/2D 
       Resistivity array in :math:`\Omega.m`. If array is two-dimensional, 
       it should be 2x2 array (real). 
       
    phi: ArrayLike 1D/2D 
       Phase array in degree (:math:`\degree`). If array is two-dimensional, 
       it should be 2x2 array (real). 

    freq: float, arraylike 1d  
      Frequency in Hz 
    
    Returns 
    ---------
    Z: Arraylike 1d or 2d , complex 
       
      Z dimension depends to the inputs array `rho` and `phi`. 

    Examples 
    ---------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import rhophi2z 
    >>> rhophi2z (823 , 25 , 500 )
    array([1300.00682824+606.20313966j])
    >>> rho = np.array ([[823, 700], [723, 526]] )
    >>> phi = np.array ([[45, 50], [90, 180]]) 
    >>> rhophi2z (rho, phi , freq= 500  ) 
    array([[ 1.01427314e+03+1.01427314e+03j,  8.50328081e+02+1.01338154e+03j],
           [ 8.23227764e-14+1.34443297e+03j, -1.14673449e+03+1.40434473e-13j]])
    >>> rhophi2z (np.array ( [ 823, 700])  , np.array ([45, 50 ])  , [500, 700] )
    array([1014.27313876+1014.27313876j, 1006.12175325+1199.04921402j])
    >>> rho  = np.abs (np.random.randn (7, 3 ) * 100 )
    >>> phi = np.abs ( np.random.randn (7, 3 ) *180 % 90 ) 
    >>> freq = np.abs ( np.random.randn (7) * 100 )
    >>> rhophi2z (rho   , phi  , freq )
    
    """
    def _rhophi2z (r, p, f ): 
        """ An isolated part of `rhophi2z """
        abs_z  = np.sqrt(5 * f * r)
        return cmath.rect(abs_z , radians(p))
    
    is_array2x2 =False 

    rho = np.array ( 
        is_iterable(rho, exclude_string= True ,
                    transform =True )) 
    phi = np.array (
        is_iterable(phi, exclude_string= True , 
                    transform =True )) 
    freq = np.array (
        is_iterable(freq, exclude_string= True , 
                    transform =True )) 

    if ( rho.shape == (2,2) or  phi.shape == (2,2)): 
        n=None 
        if rho.shape != (2,2): 
            n, t  ='Resistivity', rho
        elif phi.shape != (2,2): 
            n , t ='Phase', phi
        if n is not None: 
            raise EMError ("Resistivity and Phase must be consistent."
                           f" Expect 2 x2 array for {n}. Got {t.shape}")
            
        is_array2x2 = True 
    if not ( _is_numeric_dtype(rho) and _is_numeric_dtype(phi)): 
        raise EMError ('Resistivity and Phase arguments must be one (1D) or'
                       ' two dimensional (2x2) arrays (real)') 

    if is_array2x2 : 
        z = np.zeros((2,2),'complex')
        for i in range(2):
            for j in range(2):
                z[i, j ] = _rhophi2z(r = rho[i,j], p = phi[i,j], f = freq )
                # abs_z  = np.sqrt(5 * freq * rho[i,j])
                # z[i,j] = cmath.rect(abs_z , radians(phi[i,j]))
        return z 
    
    check_consistency_size(rho, phi, freq )
    
    if _is_arraylike_1d (phi ): 
        
        z = np.zeros_like ( phi , dtype ='complex')
        # when scalar is passed or 1d array is 
        # given 
        for ii in range ( len(phi)): #
            z [ii] = _rhophi2z ( rho[ii], phi[ii], freq[:, None ][ii] )    
    else:
        # when non square matrix is given 
        # range like freq and n_stations 
        
        z = np.zeros(( len( freq), phi.shape [1]), dtype = 'complex')
        for i in range (len(freq)): 
            for j in range(phi.shape[1]) : 
                z[i, j ] =  _rhophi2z(rho[i, j], phi[i,j], freq[i] ) 

    return z 

def z2rhoa (
        z:NDArray [DType[complex]], 
        freq: ArrayLike[DType[float]]
)-> NDArray[DType[float]]:
    r""" Convert impendance tensor z  to apparent resistivity
    
    :param z: Impedance tensor  in :math:`\Omega` 
    :type z: ndarray, shape (N, M) 
 
    :param freq: Frequency in Hertz
    :type freq: array-like , shape (N, )
    :
    :return: Apparent resistivity in :math:`\Omega.m`  
    :rtype: ndarray, shape (N, M) 
    
    :example: 
    >>> import numpy as np 
    >>> z = np.array([2 + 1j *3 ])
    >>> f = np.array ([1014])
    >>> z2rhoa(z, f)
    ... array([[1623.73691735]])
        
    """

    z = np.array(z, dtype = 'complex' ) ;  freq = np.array(freq)

    if len(freq) != len(z): 
        raise ValueError("frequency and tensor z must have the same length."
                         f"{len(freq)} & {len(z)} are given.")
 
    return np.abs(z)**2 / (2 * np.pi * freq[:, None] * mu0 )

def savitzky_golay1d (
        y: ArrayLike[DType[T]], 
        window_size:int , 
        order: int, 
        deriv: int =0, 
        rate: int =1, 
        mode: str ='same'
        )-> ArrayLike[DType[T]]:
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    
    The Savitzky-Golay filter removes high frequency noise from data. It has the 
    advantage of preserving the original shape and features of the signal better
    than other types of filtering approaches, such as moving averages techniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    mode: str 
         mode of the border prepending. Should be ``valid`` or ``same``. 
         ``same`` is used for prepending or appending the first value of
         array for smoothing.Default is ``same``.  
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly suited for 
    smoothing noisy data. The main idea behind this approach is to make for 
    each point a least-square fit with a polynomial of high order over a
    odd-sized window centered at the point.
    
    Examples
    --------
    >>> import numpy as np 
    >>> import matplotlib.pyplot as plt 
    >>> from hlearn.utils.exmath import savitzky_golay1d 
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    >>> ysg = savitzky_golay1d(y, window_size=31, order=4)
    >>> plt.plot(t, y, label='Noisy signal')
    >>> plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    >>> plt.plot(t, ysg, 'r', label='Filtered signal')
    >>> plt.legend()
    >>> plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    .. [3] https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter#Moving_average
    
    """

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    
    y = check_y( y, y_numeric= True )
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode=mode)

def interpolate2d (
        arr2d: NDArray[float] , 
        method:str  = 'slinear', 
        **kws): 
    """ Interpolate the data in 2D dimensional array. 
    
    If the data contains some missing values. It should be replaced by the 
    interpolated values. 
    
    Parameters 
    -----------
    arr2d : np.ndarray, shape  (N, M)
        2D dimensional data 
        
    method: str, default ``linear``
        Interpolation technique to use. Can be ``nearest``or ``pad``. 
    
    kws: dict 
        Additional keywords. Refer to :func:`~.interpolate1d`. 
        
    Returns 
    -------
    arr2d:  np.ndarray, shape  (N, M)
        2D dimensional data interpolated 
    
    Examples 
    ---------
    >>> from hlearn.methods.em import EM 
    >>> from hlearn.utils.exmath import interpolate2d 
    >>> # make 2d matrix of frequency
    >>> emObj = EM().fit(r'data/edis')
    >>> freq2d = emObj.make2d (out = 'freq')
    >>> freq2d_i = interpolate2d(freq2d ) 
    >>> freq2d.shape 
    ...(55, 3)
    >>> freq2d 
    ... array([[7.00000e+04, 7.00000e+04, 7.00000e+04],
           [5.88000e+04, 5.88000e+04, 5.88000e+04],
           ...
            [6.87500e+00, 6.87500e+00, 6.87500e+00],
            [        nan,         nan, 5.62500e+00]])
    >>> freq2d_i
    ... array([[7.000000e+04, 7.000000e+04, 7.000000e+04],
           [5.880000e+04, 5.880000e+04, 5.880000e+04],
           ...
           [6.875000e+00, 6.875000e+00, 6.875000e+00],
           [5.625000e+00, 5.625000e+00, 5.625000e+00]])
    
    References 
    ----------
    
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp2d.html        
        
    """ 
    arr2d = np.array(arr2d)
    
    if len(arr2d.shape) ==1: 
        arr2d = arr2d[:, None] # put on 
    if arr2d.shape[0] ==1: 
        arr2d = reshape (arr2d, axis=0)
    
    if not hasattr (arr2d , '__complex__'): 
        arr2d = check_array(
            arr2d, 
            to_frame = False, 
            input_name ="arr2d",
            force_all_finite="allow-nan" ,
            dtype =arr2d.dtype, 
            )
    arr2d  = np.hstack ([ 
        reshape (interpolate1d(arr2d[:, ii], 
                kind=method, 
                method ='pd', 
                 **kws), 
                 axis=0)
             for ii in  range (arr2d.shape[1])]
        )
    return arr2d 

def dummy_basement_curve(
        func: F ,
        ks: float ,
        slope: float | int = 45, 
)-> Tuple[F, float]: 
    """ Compute the pseudodepth from the search zone. 
    
    :param f: callable - Polyfit1D function 
    :param mz: array-zone - Expected Zone for groundwater search 
    :param ks: float - The depth from which the expected fracture 
        zone must starting looking for groundwater. 
    :param slope: float - Degree angle for slope in linear function 
        of the dummy curve
    :returns: 
        - lambda function of basement curve `func45` 
        - beta is intercept value compute for keysearch `ks`
    """
    # Use kesearch (ks) to compute the beta value from the function f
    beta = func(ks)
    # note that 45 degree is used as the slope of the 
    # imaginary basement curve
    # fdummy (x) = slope (45degree) * x + intercept (beta)
    slope = np.sin(np.deg2rad(slope))
    func45 = lambda x: slope * x + beta 
    
    return func45, beta 

def find_limit_for_integration(
        ix_arr: ArrayLike[DType[int]],
        b0: List[T] =[]
)-> List[T]: 
    r""" Use the roots between f curve and basement curves to 
    detect the limit of integration.
    
    :param ix_arr: array-like - Indexes array from masked array where  
        the value are true i.e. where :math:` b-f > 0 \Rightarrow  b> f` . 
        
    :param b0: list - Empy list to hold the limit during entire loop 
    
    .. note::
        :math:`b > f \Longrightarrow` Curve b (basement) is above the fitting  
        curve :math:`f` . :math:`b < f` otherwise. The pseudoarea is the area 
        where :math:` b > f` .
    
    :return: list - integration bounds 
    
    """
    
    s = ix_arr.min() - 1 # 0 -1 =-1
    oc = ix_arr.min() 
    for jj,  v in enumerate(ix_arr): 
        s = v - s
        if s !=1: 
            b0.append(oc); b0.append(ix_arr[jj-1])
            oc= v
        s= v 
    if v ==ix_arr[-1]: 
        b0.append(oc); b0.append(v)
        
    return b0 

def find_bound_for_integration(
        ix_arr: ArrayLike[DType[int]],
        b0: List[T] =[]
)-> List[T]: 
    r""" Recursive function to find the roots between f curve and basement 
    curves so to detect the  integration bounds. 
    
    The function use entirely numpy for seaching integration bound. 
    Since it is much faster than :func:`find_limit_for_integration` although 
    both did the same tasks. 
    
    :param ix_arr: array-like - Indexes array from masked array where 
        the value are true i.e. where :math:`b-f > 0 \Rightarrow b > f` . 
        
    :param b0: list - Empy list to hold the limit during entire loop 
    
    :return: list - integration bounds
    
    .. note::
        :math:`b > f \Longrightarrow` Curve b (basement) is above the fitting curve 
        :math:`f` . :math:`b < f` otherwise. The pseudoarea is the area where 
        :math:`b > f` .
    
    """
    
    # get the first index and arange this thin the end 
    psdiff = np.arange(ix_arr.min(), len(ix_arr) + ix_arr.min(), 1) 
    # make the difference to find the zeros values 
    diff = ix_arr - psdiff 
    index, = np.where(diff ==0) ; 
    # take the min index and max index 
    b0.append(min(ix_arr[index]))
    b0.append(max(ix_arr[index]))
    #now take the max index and add +1 and start by this part 
    # retreived the values 
    array_init = ix_arr[int(max(index)) +1:]
    return b0 if len(
        array_init)==0 else find_bound_for_integration(array_init, b0)
    
def fitfunc(
        x: ArrayLike[T], 
        y: ArrayLike[T], 
        deg: float | int  =None,
        sample: int =1000
)-> Tuple[F, ArrayLike[T]]: 
    """ Create polyfit function from a specifc sample data points. 
    
    :param x: array-like of x-axis.
    
    :param y: array-like of y-axis.
    
    :param deg: polynomial degree. If ``None`` should compute using the 
        length of  extrema (local + global).
        
    :param sample: int - Number of data points should use for fitting 
        function. Default is ``1000``. 
    
    :returns: 
        - Polynomial function `f` 
        - new axis  `x_new` generated from the samples.
        - projected sample values got from `f`.
    """
    for ar, n in  zip ((x, y),("x", "y")): 
        if not _is_arraylike_1d(ar): 
            raise TypeError (f"{n!r} only supports 1d array.")
    # generate a sample of values to cover the fit function 
    # thus compute ynew (yn) from the poly function f
    minl, = argrelextrema(y, np.less) 
    maxl, = argrelextrema(y,np.greater)
    # get the number of degrees
    degree = len(minl) + len(maxl)

    coeff = np.polyfit(x, y, deg if deg is not None else degree + 1 )
    f = np.poly1d(coeff)
    xn = np.linspace(min(x), max(x), sample)
    yp = f(xn)
    
    return f, xn, yp  

def vesDataOperator(
        AB : ArrayLike = None, 
        rhoa: ArrayLike= None ,
        data: DataFrame  =None,
        typeofop: str = None, 
        outdf: bool = False, 
)-> Tuple[ArrayLike] | DataFrame : 
    """ Check the data in the given deep measurement and set the suitable
    operations for duplicated spacing distance of current electrodes `AB`. 
    
    Sometimes at the potential electrodes (`MN`), the measurement of `AB` are 
    collected twice after modifying the distance of `MN` a bit. At this point, 
    two or many resistivity values are targetted to the same distance `AB`  
    (`AB` still remains unchangeable while while `MN` is changed). So the 
    operation consists whether to average (``mean``) the resistiviy values or 
    to take the ``median`` values or to ``leaveOneOut`` (i.e. keep one value
    of resistivity among the different values collected at the same point`AB`)
    at the same spacing `AB`. Note that for the `LeaveOneOut``, the selected 
    resistivity value is randomly chosen. 
    
    Parameters 
    -----------
    AB: array-like 1d, 
        Spacing of the current electrodes when exploring in deeper. 
        Is the depth measurement (AB/2) using the current electrodes AB.
        Units are in meters. 
    
    rhoa: array-like 1d
        Apparent resistivity values collected by imaging in depth. 
        Units are in :math:`\Omega {.m}` not :math:`log10(\Omega {.m})`
    
    data: DataFrame, 
        It is composed of spacing values `AB` and  the apparent resistivity 
        values `rhoa`. If `data` is given, params `AB` and `rhoa` should be 
        kept to ``None``.   
    
    typeofop: str,['mean'|'median'|'leaveoneout'], default='mean' 
        Type of operation to apply  to the resistivity values `rhoa` of the 
        duplicated spacing points `AB`. The default operation is ``mean``. 
    
    outdf: bool , default=False, 
        Outpout a new dataframe composed of `AB` and `rhoa`; data renewed. 
    
    Returns 
    ---------
        - Tuple of (AB, rhoa): New values computed from `typeofop` 
        - DataFrame: New dataframe outputed only if ``outdf`` is ``True``.
        
    Notes 
    ---------
    By convention `AB` and `MN` are half-space dipole length which 
    correspond to `AB/2` and `MN/2` respectively. 
    
    Examples 
    ---------
    >>> from hlearn.utils.exmath import vesDataOperator
    >>> from hlearn.utils.coreutils import vesSelector 
    >>> data = vesSelector ('data/ves/ves_gbalo.xlsx')
    >>> len(data)
    ... (32, 3) # include the potentiel electrode values `MN`
    >>> df= vesDataOperator(data.AB, data.resistivity,
                            typeofop='leaveOneOut', outdf =True)
    >>> df.shape 
    ... (26, 2) # exclude `MN` values and reduce(-6) the duplicated values. 
    """
    op = copy.deepcopy(typeofop) 
    typeofop= str(typeofop).lower()
    if typeofop not in ('none', 'mean', 'median', 'leaveoneout'):
        raise ValueError(
            f'Unacceptable argument {op!r}. Use one of the following '
            f'argument {smart_format([None,"mean", "median", "leaveOneOut"])}'
            ' instead.')

    typeofop ='mean' if typeofop =='none' else typeofop 
    
    AB, rhoa = _validate_ves_operator(
        AB, rhoa, data = data , exception= VESError )

    #----> When exploring in deeper, after changing the distance 
    # of MN , measure are repeated at the same points. So, we will 
    # selected these points and take the mean values of tyhe resistivity         
    # make copies 
    AB_ = AB.copy() ; rhoa_= rhoa.copy() 
    # find the duplicated values 
    # with np.errstate(all='ignore'):
    mask = np.zeros_like (AB_, dtype =bool) 
    mask[np.unique(AB_, return_index =True)[1]]=True 
    dup_values = AB_[~mask]
    
    indexes, = np.where(AB_==dup_values)
    #make a copy of unique values and filled the duplicated
    # values by their corresponding mean resistivity values 
    X, rindex  = np.unique (AB_, return_index=True); Y = rhoa_[rindex]
    d0= np.zeros_like(dup_values)
    for ii, d in enumerate(dup_values): 
       index, =  np.where (AB_==d)
       if typeofop =='mean': 
           d0[ii] = rhoa_[index].mean() 
       elif typeofop =='median': 
           d0[ii] = np.median(rhoa_[index])
       elif typeofop =='leaveoneout': 
           d0[ii] = np.random.permutation(rhoa_[index])[0]
      
    maskr = np.isin(X, dup_values, assume_unique=True)
    Y[maskr] = d0
    
    return (X, Y) if not outdf else pd.DataFrame (
        {'AB': X,'resistivity':Y}, index =range(len(X)))

 
@refAppender(refglossary.__doc__)
@docSanitizer()    
def scalePosition(
        ydata: ArrayLike | SP | Series | DataFrame ,
        xdata: ArrayLike| Series = None, 
        func : Optional [F] = None ,
        c_order: Optional[int|str] = 0,
        show: bool =False, 
        **kws): 
    """ Correct data location or position and return new corrected location 
    
    Parameters 
    ----------
    ydata: array_like, series or dataframe
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
        
    xdata: array_like or object
        The independent variable where the data is measured. Should usually 
        be an M-length sequence or an (k,M)-shaped array for functions with
        k predictors, but can actually be any object. If ``None``, `xdata` is 
        generated by default using the length of the given `ydata`.
        
    func: callable 
        The model function, ``f(x, ...)``. It must take the independent variable 
        as the first argument and the parameters to fit as separate remaining
        arguments. The default `func` is ``linear`` function i.e  for ``f(x)= ax +b``. 
        where `a` is slope and `b` is the intercept value. Setting your own 
        function for better fitting is recommended. 
        
    c_order: int or str
        The index or the column name if ``ydata`` is given as a dataframe to 
        select the right column for scaling.
    show: bool 
        Quick visualization of data distribution. 

    kws: dict 
        Additional keyword argument from  `scipy.optimize_curvefit` parameters. 
        Refer to `scipy.optimize.curve_fit`_.  
        
    Returns 
    --------
    - ydata - array -like - Data scaled 
    - popt - array-like Optimal values for the parameters so that the sum of 
        the squared residuals of ``f(xdata, *popt) - ydata`` is minimized.
    - pcov - array like The estimated covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard deviation 
        errors on the parameters use ``perr = np.sqrt(np.diag(pcov))``. How the
        sigma parameter affects the estimated covariance depends on absolute_sigma 
        argument, as described above. If the Jacobian matrix at the solution
        doesn’t have a full rank, then ‘lm’ method returns a matrix filled with
        np.inf, on the other hand 'trf' and 'dogbox' methods use Moore-Penrose
        pseudoinverse to compute the covariance matrix.
        
    Examples
    --------
    >>> from hlearn.utils import erpSelector, scalePosition 
    >>> df = erpSelector('data/erp/l10_gbalo.xlsx') 
    >>> df.columns 
    ... Index(['station', 'resistivity', 'longitude', 'latitude', 'easting',
           'northing'],
          dtype='object')
    >>> # correcting northing coordinates from easting data 
    >>> northing_corrected, popt, pcov = scalePosition(ydata =df.northing , 
                                               xdata = df.easting, show=True)
    >>> len(df.northing.values) , len(northing_corrected)
    ... (20, 20)
    >>> popt  # by default popt =(slope:a ,intercept: b)
    ...  array([1.01151734e+00, 2.93731377e+05])
    >>> # corrected easting coordinates using the default x.
    >>> easting_corrected, *_= scalePosition(ydata =df.easting , show=True)
    >>> df.easting.values 
    ... array([790284, 790281, 790277, 790270, 790265, 790260, 790254, 790248,
    ...       790243, 790237, 790231, 790224, 790218, 790211, 790206, 790200,
    ...       790194, 790187, 790181, 790175], dtype=int64)
    >>> easting_corrected
    ... array([790288.18571705, 790282.30300999, 790276.42030293, 790270.53759587,
    ...       790264.6548888 , 790258.77218174, 790252.88947468, 790247.00676762,
    ...       790241.12406056, 790235.2413535 , 790229.35864644, 790223.47593938,
    ...       790217.59323232, 790211.71052526, 790205.8278182 , 790199.94511114,
    ...       790194.06240407, 790188.17969701, 790182.29698995, 790176.41428289])
    
    """
    def linfunc (x, a, b): 
        """ Set the simple linear function"""
        return a * x + b 
        
    if str(func).lower() in ('none' , 'linear'): 
        func = linfunc 
    elif not hasattr(func, '__call__') or not inspect.isfunction (func): 
        raise TypeError(
            f'`func` argument is a callable not {type(func).__name__!r}')
        
    ydata = _assert_all_types(ydata, list, tuple, np.ndarray,
                              pd.Series, pd.DataFrame  )
    c_order = _assert_all_types(c_order, int, float, str)
    try : c_order = int(c_order) 
    except: pass 

    if isinstance(ydata, pd.DataFrame): 
        if c_order ==0: 
            warnings.warn("The first column of the data should be considered"
                          " as the `y` target.")
        if c_order is None: 
            raise TypeError('Dataframe is given. The `c_order` argument should '
                            'be defined for column selection. Use column name'
                            ' instead')
        if isinstance(c_order, str): 
            # check whether the value is on the column name
            if c_order.lower() not in list(map( 
                    lambda x :x.lower(), ydata.columns)): 
                raise ValueError (
                    f'c_order {c_order!r} not found in {list(ydata.columns)}'
                    ' Use the index instead.')
                # if c_order exists find the index and get the 
                # right column name 
            ix_c = list(map( lambda x :x.lower(), ydata.columns)
                        ).index(c_order.lower())
            ydata = ydata.iloc [:, ix_c] # series 
        elif isinstance (c_order, (int, float)): 
            c_order =int(c_order) 
            if c_order >= len(ydata.columns): 
                raise ValueError(
                    f"`c_order`'{c_order}' should be less than the number of " 
                    f"given columns '{len(ydata.columns)}'. Use column name instead.")
            ydata= ydata.iloc[:, c_order]
                  
    ydata = check_y (np.array(ydata)  , input_name= "ydata")
    
    if xdata is None: 
        xdata = np.linspace(0, 4, len(ydata))
        
    xdata = check_y (xdata , input_name= "Xdata")
    
    if len(xdata) != len(ydata): 
        raise ValueError(" `x` and `y` arrays must have the same length."
                        "'{len(xdata)}' and '{len(ydata)}' are given.")
        
    popt, pcov = curve_fit(func, xdata, ydata, **kws)
    ydata_new = func(xdata, *popt)
    
    if show:
        plt.plot(xdata, ydata, 'b-', label='data')
        plt.plot(xdata, func(xdata, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        
    return ydata_new, popt, pcov 

def __sves__ (
        s_index: int  , 
        cz: ArrayLike | List[float], 
) -> Tuple[ArrayLike, ArrayLike]: 
    """ Divide the conductive zone in leftzone and rightzone from 
    the drilling location index . 

    :param s_index - station location index expected for dilling location. 
        It refers to the position of |VES|. 
        
    :param cz: array-like - Conductive zone . 
    
    :returns: 
        - <--Sves: Left side of conductive zone from |VES| location. 
        - --> Sves: Right side of conductive zone from |VES| location. 
        
    .. note:: Both sides included the  |VES| `Sves` position.
    .. |VES| replace:: Vertical Electrical Sounding 
    """
    try:  s_index = int(s_index)
    except: raise TypeError(
        f'Expected integer value not {type(s_index).__name__}')
    
    s_index = _assert_all_types( s_index , int)
    cz = _assert_all_types(cz, np.ndarray, pd.Series, list, tuple )

    rmax_ls , rmax_rs = max(cz[:s_index  + 1]), max(cz[s_index  :]) 
    # detect the value of rho max  (rmax_...) 
    # from lower side bound of the anomaly.
    rho_ls= rmax_ls if rmax_ls  <  rmax_rs else rmax_rs 
    
    side =... 
    # find with positions 
    for _, sid  in zip((rmax_ls , rmax_rs ) , ('leftside', 'rightside')) : 
            side = sid ; break 
        
    return (rho_ls, side), (rmax_ls , rmax_rs )

def detect_station_position (
        s : Union[str, int] ,
        p: SP, 
) -> Tuple [int, float]: 
    """ Detect station position and return the index in positions
    
    :param s: str, int - Station location  in the position array. It should 
        be the positionning of the drilling location. If the value given
        is type string. It should be match the exact position to 
        locate the drilling. Otherwise, if the value given is in float or 
        integer type, it should be match the index of the position array. 
         
    :param p: Array-like - Should be the  conductive zone as array of 
        station location values. 
            
    :returns: 
        - `s_index`- the position index location in the conductive zone.  
        - `s`- the station position in distance. 
        
    :Example: 
        
        >>> import numpy as np 
        >>> from hlearn.utils.exmath import detect_station_position 
        >>> pos = np.arange(0 , 50 , 10 )
        >>> detect_station_position (s ='S30', p = pos)
        ... (3, 30.0)
        >>> detect_station_position (s ='40', p = pos)
        ... (4, 40.0)
        >>> detect_station_position (s =2, p = pos)
        ... (2, 20)
        >>> detect_station_position (s ='sta200', p = pos)
        ... hlearnError_station: Station sta200 \
            is out of the range; max position = 40
    """
    s = _assert_all_types( s, float, int, str)
    
    p = check_y (p, input_name ="Position array 'p'", to_frame =True )
    
    S=copy.deepcopy(s)
    if isinstance(s, str): 
        s =s.lower().replace('s', '').replace('pk', '').replace('ta', '')
        try : 
            s=int(s)
        except : 
            raise ValueError (f'could not convert string to float: {S}')
            
    p = np.array(p, dtype = np.int32)
    dl = (p.max() - p.min() ) / (len(p) -1) 
    if isinstance(s, (int, float)): 
        if s > len(p): # consider this as the dipole length position: 
            # now let check whether the given value is module of the station 
            if s % dl !=0 : 
                raise StationError  (
                    f'Unable to detect the station position {S}')
            elif s % dl == 0 and s <= p.max(): 
                # take the index 
                s_index = s//dl
                return int(s_index), s_index * dl 
            else : 
                raise StationError (
                    f'Station {S} is out of the range; max position = {max(p)}'
                )
        else : 
            if s >= len(p): 
                raise StationError (
                    'Location index must be less than the number of'
                    f' stations = {len(p)}. {s} is gotten.')
            # consider it as integer index 
            # erase the last variable
            # s_index = s 
            # s = S * dl   # find 
            return s , p[s ]
       
    # check whether the s value is in the p 
    if True in np.isin (p, s): 
        s_index ,  = np.where (p ==s ) 
        s = p [s_index]
        
    return int(s_index) , s 


def _manage_colors (c, default = ['ok', 'ob-', 'r-']): 
    """ Manage the ohmic-area plot colors """
    c = c or default 
    if isinstance(c, str): 
        c= [c] 
    c = list(c) +  default 
    
    return c [:3] # return 3colors 
     

   
def quickplot (arr: ArrayLike | List[float], dl:float  =10)-> None: 
    """Quick plot to see the anomaly"""
    
    plt.plot(np.arange(0, len(arr) * dl, dl), arr , ls ='-', c='k')
    plt.show() 
 
def magnitude (cz:Sub[ArrayLike[float, DType[float]]] ) -> float: 
    r""" 
    Compute the magnitude of selected conductive zone. 
    
    The magnitude parameter is the absolute resistivity value between
    the minimum :math:`\min \rho_a` and maximum :math:`\max \rho_a` 
    value of selected anomaly:
    
    .. math::
    
        magnitude=|\min\rho_a -\max\rho_a|

    :param cz: array-like. Array of apparent resistivity values composing 
        the conductive zone. 
    
    :return: Absolute value of anomaly magnitude in ohm.meters.
    """
    return np.abs (cz.max()- cz.min()) 

def power (p:Sub[SP[ArrayLike, DType [int]]] | List[int] ) -> float : 
    """ 
    Compute the power of the selected conductive zone. Anomaly `power` 
    is closely referred to the width of the conductive zone.
    
    The power parameter implicitly defines the width of the conductive zone
    and is evaluated from the difference between the abscissa 
    :math:`X_{LB}` and the end :math:`X_{UB}` points of 
    the selected anomaly:
    
    .. math::
        
        power=|X_{LB} - X_{UB} |
    
    :param p: array-like. Station position of conductive zone.
    
    :return: Absolute value of the width of conductive zone in meters. 
    
    """
    return np.abs(p.min()- p.max()) 

def _find_cz_bound_indexes (
    erp: Union[ArrayLike[float, DType[float]], List[float], pd.Series],
    cz: Union [Sub[ArrayLike], List[float]] 
)-> Tuple[int, int]: 
    """ 
    Fetch the limits 'LB' and 'UB' of the selected conductive zone.
    
    Indeed the 'LB' and 'UB' fit the lower and upper boundaries of the 
    conductive zone respectively. 
    
    :param erp: array-like. Apparent resistivities collected during the survey. 
    :param cz: array-like. Array of apparent resistivies composing the  
        conductive zone. 
    
    :return: The index of boundaries 'LB' and 'UB'. 
    
    .. note::
        
        `cz` must be self-containing of `erp`. If ``False`` should raise and error. 
        
    """
    # assert whether cz is a subset of erp. 
    if isinstance( erp, pd.Series): erp = erp.values 

    if not np.isin(True,  (np.isin (erp, cz))):
        raise ValueError ('Expected the conductive zone array being a '
                          'subset of the resistivity array.')
    # find the indexes using np.argwhere  
    cz_indexes = np.argwhere(np.isin(erp, cz)).ravel()
    
    return cz_indexes [0] , cz_indexes [-1] 

def convert_distance_to_m(
        value:T ,
        converter:float =1e3,
        unit:str ='km'
)-> float: 
    """ Convert distance from `km` to `m` or vice versa even a string 
    value is given.
    
    :param value: value to convert. 
    :paramm converter: Equivalent if given in ``km`` rather than ``m``.
    :param unit: unit to convert to.
    
    """
    
    if isinstance(value, str): 
        try:
            value = float(value.replace(unit, '')
                              )*converter if value.find(
                'km')>=0 else float(value.replace('m', ''))
        except: 
            raise TypeError(f"Expected float not {type(value)!r}."
               )
            
    return value
       
def get_station_number (
        dipole:float,
        distance:float , 
        from0:bool = False,
        **kws
)-> float: 
    """ Get the station number from dipole length and 
    the distance to the station.
    
    :param distance: Is the distance from the first station to `s` in 
        meter (m). If value is given, please specify the dipole length in 
        the same unit as `distance`.
    :param dipole: Is the distance of the dipole measurement. 
        By default the dipole length is in meter.
    :param kws: :func:`convert_distance_to_m` additional arguments
    
    """
    
    dipole=convert_distance_to_m(dipole, **kws)
    distance =convert_distance_to_m(distance, **kws)

    return  distance/dipole  if from0 else distance/dipole + 1 

@deprecated('Function is going to be removed for the next release ...')
def define_conductive_zone (
        erp: ArrayLike | List[float],
        stn: Optional [int] = None,
        sres:Optional [float] = None,
        *, 
        distance:float | None = None , 
        dipole_length:float | None = None,
        extent:int =7): 
    """ Detect the conductive zone from `s`ves point.
    
    :param erp: Resistivity values of electrical resistivity profiling(ERP).
    
    :param stn: Station number expected for VES and/or drilling location.
    
    :param sres: Resistivity value at station number `stn`. 
        If `sres` is given, the auto-search will be triggered to 
        find the station number that fits the resistivity value. 
    
    :param distance: Distance from the first station to `stn`. If given, 
        be sure to provide the `dipole_length`
    :param dipole_length: Length of the dipole. Comonly the distante between 
        two close stations. Since we use config AB/2 
    :param extent: Is the width to depict the anomaly. If provide, need to be 
        consistent along all ERP line. Should keep unchanged for other 
        parameters definitions. Default is ``7``.
    :returns: 
        - CZ:Conductive zone including the station position 
        - sres: Resistivity value of the station number
        - ix_stn: Station position in the CZ
            
    .. note:: 
        If many stations got the same `sres` value, the first station 
        is flagged. This may not correspond to the station number that is 
        searching. Use `sres` only if you are sure that the 
        resistivity value is unique on the whole ERP. Otherwise it's 
        not recommended.
        
    :Example: 
        >>> import numpy as np
        >>> from hlearn.utils.exmath import define_conductive_zone 
        >>> sample = np.random.randn(9)
        >>> cz, stn_res = define_conductive_zone(sample, 4, extent = 7)
        ... (array([ 0.32208638,  1.48349508,  0.6871188 , -0.96007639,
                    -1.08735204,0.79811492, -0.31216716]),
             -0.9600763919368086, 
             3)
    """
    try : 
        iter(erp)
    except : raise ERPError (
            f'`erp` must be a sequence of values not {type(erp)!r}')
    finally: erp = np.array(erp)
  
    # check the distance 
    if stn is None: 
        if (dipole_length and distance) is not None: 
            stn = get_station_number(dipole_length, distance)
        elif sres is not None: 
            snix, = np.where(erp==sres)
            if len(snix)==0: 
                raise VESError(
                    "Could not  find the resistivity value of the VES "
                    "station. Please provide the right value instead.") 
                
            elif len(snix)==2: 
                stn = int(snix[0]) + 1
        else :
            raise StationError (
                '`stn` is needed or at least provide the survey '
                'dipole length and the distance from the first '
                'station to the VES station. ')
            
    if erp.size < stn : 
        raise StationError(
            f"Wrong station number =`{stn}`. Is larger than the "
            f" number of ERP stations = `{erp.size}` ")
    
    # now defined the anomaly boundaries from sn
    stn =  1 if stn == 0 else stn  
    stn -=1 # start counting from 0.
    if extent %2 ==0: 
        if len(erp[:stn]) > len(erp[stn:])-1:
           ub = erp[stn:][:extent//2 +1]
           lb = erp[:stn][len(ub)-int(extent):]
        elif len(erp[:stn]) < len(erp[stn:])-1:
            lb = erp[:stn][stn-extent//2 +1:stn]
            ub= erp[stn:][:int(extent)- len(lb)]
     
    else : 
        lb = erp[:stn][-extent//2:] 
        ub = erp[stn:][:int(extent//2)+ 1]
    
    # read this part if extent anomaly is not reached
    if len(ub) +len(lb) < extent: 
        if len(erp[:stn]) > len(erp[stn:])-1:
            add = abs(len(ub)-len(lb)) # remain value to add 
            lb = erp[:stn][-add -len(lb) - 1:]
        elif len(erp[:stn]) < len(erp[stn:])-1:
            add = abs(len(ub)-len(lb)) # remain value to add 
            ub = erp[stn:][:len(ub)+ add -1] 
          
    conductive_zone = np.concatenate((lb, ub))
    # get the index of station number from the conductive zone.
    ix_stn, = np.where (conductive_zone == conductive_zone[stn])
    ix_stn = int(ix_stn[0]) if len(ix_stn)> 1 else  int(ix_stn)
    
    return  conductive_zone, conductive_zone[stn], ix_stn 
    
#FR0: #CED9EF # (206, 217, 239)
#FR1: #9EB3DD # (158, 179, 221)
#FR2: #3B70F2 # (59, 112, 242) #repl rgb(52, 54, 99)
#FR3: #0A4CEE # (10, 76, 238)

def shortPlot (erp, cz=None): 
    """ 
    Quick plot to visualize the `sample` of ERP data overlained to the  
    selected conductive zone if given.
    
    :param erp: array_like, the electrical profiling array 
    :param cz: array_like, the selected conductive zone. If ``None``, `cz` 
        should be plotted.
    
    :Example: 
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import shortPlot, define_conductive_zone 
    >>> test_array = np.random.randn (10)
    >>> selected_cz ,*_ = define_conductive_zone(test_array, 7) 
    >>> shortPlot(test_array, selected_cz )
        
    """
    erp = check_y (erp , input_name ="sample of ERP data")
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots(1,1, figsize =(10, 4))
    leg =[]
    ax.scatter (np.arange(len(erp)), erp, marker ='.', c='b')
    zl, = ax.plot(np.arange(len(erp)), erp, 
                  c='r', 
                  label ='Electrical resistivity profiling')
    leg.append(zl)
    if cz is not None: 
        cz= check_y (cz, input_name ="Conductive zone 'cz'")
        # construct a mask array with np.isin to check whether 
        # `cz` is subset array
        z = np.ma.masked_values (erp, np.isin(erp, cz ))
        # a masked value is constructed so we need 
        # to get the attribute fill_value as a mask 
        # However, we need to use np.invert or tilde operator  
        # to specify that other value except the `CZ` values mus be 
        # masked. Note that the dtype must be changed to boolean
        sample_masked = np.ma.array(
            erp, mask = ~z.fill_value.astype('bool') )
    
        czl, = ax.plot(
            np.arange(len(erp)), sample_masked, 
            ls='-',
            c='#0A4CEE',
            lw =2, 
            label ='Conductive zone')
        leg.append(czl)

    ax.set_xticks(range(len(erp)))
    ax.set_xticklabels(
        ['S{0:02}'.format(i+1) for i in range(len(erp))])
    
    ax.set_xlabel('Stations')
    ax.set_ylabel('app.resistivity (ohm.m)')
    ax.legend( handles = leg, 
              loc ='best')
        
    plt.show()
    
@deprecated ('Expensive function; should be removed for the next realease.')
def compute_sfi (
        pk_min: float,
        pk_max: float, 
        rhoa_min: float,
        rhoa_max: float, 
        rhoa: ArrayLike | List[float], 
        pk: SP[int]
        ) -> float : 
    """
    SFI is introduced to evaluate the ratio of presumed existing fracture
    from anomaly extent. We use a similar approach as IF computation
    proposed by Dieng et al. (2004) to evaluate each selected anomaly 
    extent and the normal distribution of resistivity values along the 
    survey line. The SFI threshold is set at :math:`sqrt(2)`  for 
    symmetrical anomaly characterized by a perfect distribution of 
    resistivity in a homogenous medium. 
    
    :param pk_min: see :func:`compute_power` 
    :param pk_max: see :func:`compute_power` 
    
    :param rhoa_max: see :func:`compute_magnitude` 
    :param rhoa_min: see :func:`compute_magnitude`
    
    :param pk: 
        
        Station position of the selected anomaly in ``float`` value. 
        
    :param rhoa: 
        
        Selected anomaly apparent resistivity value in ohm.m 
        
    :return: standard fracture index (SFI)
    :rtype: float 
    
    :Example: 
        
        >>> from hlearn.utils.exmath import compute_sfi 
        >>> sfi = compute_sfi(pk_min = 90,
        ...                      pk_max=130,
        ...                      rhoa_min=175,
        ...                      rhoa_max=170,
        ...                      rhoa=132,
        ...                      pk=110)
        >>> sfi
    
    """  
    def deprecated_sfi_computation () : 
        """ Deprecated way for `sfi` computation"""
        try : 
            if  pk_min -pk  < pk_max - pk  : 
                sfi= np.sqrt((((rhoa_max -rhoa) / 
                                  (rhoa_min- rhoa)) **2 + 
                                 ((pk_max - pk)/(pk_min -pk))**2 ))
            elif pk_max -pk  < pk_min - pk : 
                sfi= np.sqrt((((rhoa_max -rhoa) / 
                                  (rhoa_min- rhoa)) **2 + 
                                 ((pk_min - pk)/(pk_max -pk))**2 ))
        except : 
            if sfi ==np.nan : 
                sfi = - np.sqrt(2)
            else :
                sfi = - np.sqrt(2)
       
    try : 
        
        if (rhoa == rhoa_min and pk == pk_min) or\
            (rhoa==rhoa_max and pk == pk_max): 
            ma= max([rhoa_min, rhoa_max])
            ma_star = min([rhoa_min, rhoa_max])
            pa= max([pk_min, pk_max])
            pa_star = min([pk_min, pk_max])
    
        else : 
       
            if  rhoa_min >= rhoa_max : 
                max_rho = rhoa_min
                min_rho = rhoa_max 
            elif rhoa_min < rhoa_max: 
                max_rho = rhoa_max 
                min_rho = rhoa_min 
            
            ma_star = abs(min_rho - rhoa)
            ma = abs(max_rho- rhoa )
            
            ratio = ma_star / ma 
            pa = abs(pk_min - pk_max)
            pa_star = ratio *pa
            
        sfi = np.sqrt((pa_star/ pa)**2 + (ma_star/ma)**2)
        
        if sfi ==np.nan : 
                sfi = - np.sqrt(2)
    except : 

        sfi = - np.sqrt(2)
  
    return sfi
  

def scaley(
        y: ArrayLike , 
        x: ArrayLike =None, 
        deg: int = None,  
        func:F =None
        )-> Tuple[ArrayLike, ArrayLike, F]: 
    """ Scaling value using a fitting curve. 
    
    Create polyfit function from a specifc data points `x` to correct `y` 
    values.  
    
    :param y: array-like of y-axis. Is the array of value to be scaled. 
    
    :param x: array-like of x-axis. If `x` is given, it should be the same 
        length as `y`, otherwise and error will occurs. Default is ``None``. 
    
    :param func: callable - The model function, ``f(x, ...)``. It must take 
        the independent variable as the first argument and the parameters
        to fit as separate remaining arguments.  `func` can be a ``linear``
        function i.e  for ``f(x)= ax +b`` where `a` is slope and `b` is the 
        intercept value. It is recommended according to the `y` value 
        distribution to set up  a custom function for better fitting. If `func`
        is given, the `deg` is not needed.   
        
    :param deg: polynomial degree. If  value is ``None``, it should  be 
        computed using the length of extrema (local and/or global) values.
 
    :returns: 
        - y: array scaled - projected sample values got from `f`.
        - x: new x-axis - new axis  `x_new` generated from the samples.
        - linear of polynomial function `f` 
        
    :references: 
        Wikipedia, Curve fitting, https://en.wikipedia.org/wiki/Curve_fitting
        Wikipedia, Polynomial interpolation, https://en.wikipedia.org/wiki/Polynomial_interpolation
    :Example: 
        >>> import numpy as np 
        >>> import matplotlib.pyplot as plt 
        >>> from hlearn.exmath import scale_values 
        >>> rdn = np.random.RandomState(42) 
        >>> x0 =10 * rdn.rand(50)
        >>> y = 2 * x0  +  rnd.randn(50) -1
        >>> plt.scatter(x0, y)
        >>> yc, x , f = scale_values(y) 
        >>> plt.plot(x, y, x, yc) 
        
    """   
    y = check_y( y )
    
    if str(func).lower() != 'none': 
        if not hasattr(func, '__call__') or not inspect.isfunction (func): 
            raise TypeError(
                f'`func` argument is a callable not {type(func).__name__!r}')

    # get the number of local minimum to approximate degree. 
    minl, = argrelextrema(y, np.less) 
    # get the number of degrees
    degree = len(minl) + 1
    if x is None: 
        x = np.arange(len(y)) # np.linspace(0, 4, len(y))
        
    x= check_y (x , input_name="x") 
    
    if len(x) != len(y): 
        raise ValueError(" `x` and `y` arrays must have the same length."
                        f"'{len(x)}' and '{len(y)}' are given.")
        
    coeff = np.polyfit(x, y, int(deg) if deg is not None else degree)
    f = np.poly1d(coeff) if func is  None else func 
    yc = f (x ) # corrected value of y 

    return  yc, x ,  f  

def smooth1d(
    ar, /, 
    drop_outliers:bool=True, 
    ma:bool=True, 
    absolute:bool=False, 
    view:bool=False , 
    x: ArrayLike=None, 
    xlabel:str =None, 
    ylabel:str =None, 
    fig_size:tuple = ( 10, 5) 
    )-> ArrayLike[float]: 
    """ Smooth one-dimensional array. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d 
       Array of one-dimensional 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value. This seems more 
       realistic.
       
    absolute: bool, default=False, 
       keep postive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    view: bool, default =False 
       Display curves 
    x: ArrayLike, optional 
       Abscissa array for visualization. If given, it must be consistent 
       with the given array `ar`. Raises error otherwise. 
    xlabel: str, optional 
       Label of x 
    ylabel:str, optional 
       label of y  
    fig_size: tuple , default=(10, 5)
       Matplotlib figure size
       
    Returns 
    --------
    yc: ArrayLike 
       Smoothed array value. 
       
    Examples 
    ---------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import smooth1d 
    >>> # add Guassian Noise 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20 ) * 20 + np.random.normal ( 20 )
    >>> ar [:7 ]
    array([6.42891445e+00, 3.75072493e-02, 1.82905357e+01, 2.92957265e+01,
           6.20589038e+01, 2.26399535e+01, 1.12596434e+01])
    >>> arc = smooth1d (ar, view =True , ma =False )
    >>> arc [:7 ]
    array([12.08603102, 15.29819907, 18.017749  , 20.27968322, 22.11900412,
           23.5707141 , 24.66981557])
    >>> arc = smooth1d (ar, view =True )# ma=True by default 
    array([ 5.0071604 ,  5.90839339,  9.6264018 , 13.94679804, 17.67369252,
           20.34922943, 22.00836725])
    """
    # convert data into an iterable object 
    ar = np.array(
        is_iterable(ar, exclude_string = True , transform =True )) 
    
    if not _is_arraylike_1d(ar): 
        raise TypeError("Expect one-dimensional array. Use `hlearn.smoothing`"
                        " for handling two-dimensional array.")
    if not _is_numeric_dtype(ar): 
        raise ValueError (f"{ar.dtype.name!r} is not allowed. Expect a numeric"
                          " array")
        
    arr = ar.copy() 
    if drop_outliers: 
        arr = remove_outliers( arr, fill_value = np.nan  )
    # Nan is not allow so fill NaN if exists in array 
    # is arraylike 1d 
    arr = reshape ( fillNaN( arr , method ='both') ) 
    if ma: 
        arr = moving_average(arr, method ='sma')
    # if extrapolation give negative  values
    # whether to keep as it was or convert to positive values. 
    # note that converting to positive values is 
    arr, *_  = scaley ( arr ) 
    # if extrapolation gives negative values
    # convert to positive values or keep it intact. 
    # note that converting to positive values is 
    # can be used as the last option when array 
    # data must be positive.
    if absolute: 
        arr = np.abs (arr )
    if view: 
        x = np.arange ( len(ar )) if x is None else np.array (x )

        check_consistency_size( x, ar )
            
        fig,  ax = plt.subplots (1, 1, figsize = fig_size)
        ax.plot (x, 
                 ar , 
                 'ok-', 
                 label ='raw curve'
                 )
        ax.plot (x, 
                 arr, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='smooth curve'
                 ) 
        
        ax.legend ( ) 
        ax.set_xlabel (xlabel or '')
        ax.set_ylabel ( ylabel or '') 
        
    return arr 

def smoothing (
    ar, /, 
    drop_outliers = True ,
    ma=True,
    absolute =False,
    axis = 0, 
    view = False, 
    fig_size =(7, 7), 
    xlabel =None, 
    ylabel =None , 
    cmap ='binary'
    ): 
    """ Smooth data along axis. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d or 2d 
       One dimensional or two dimensional array. 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing along the given axis 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value along axis. This seems 
       more realistic rather than using only the scaling method. 
       
    absolute: bool, default=False, 
       keep postive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    axis: int, default=0 
       Axis along with the data must be smoothed. The default is the along  
       the row. 
       
    view: bool, default =False 
       Visualize the two dimensional raw and smoothing grid. 
       
    xlabel: str, optional 
       Label of x 
       
    ylabel:str, optional 
    
       label of y  
    fig_size: tuple , default=(7, 5)
       Matplotlib figure size 
       
    cmap: str, default='binary'
       Matplotlib.colormap to manage the `view` color 
      
    Return 
    --------
    arr0: ArrayLike 
       Smoothed array value. 
    
    Examples 
    ---------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import smoothing
    >>> # add Guassian Noises 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20, 7 ) * 20 + np.random.normal ( 20, 7 )
    >>> ar [:3, :3 ]
    array([[ 31.5265026 ,  18.82693352,  34.5459903 ],
           [ 36.94091413,  12.20273182,  32.44342041],
           [-12.90613711,  10.34646896,   1.33559714]])
    >>> arc = smoothing (ar, view =True , ma =False )
    >>> arc [:3, :3 ]
    array([[32.20356863, 17.18624398, 41.22258603],
           [33.46353806, 15.56839464, 19.20963317],
           [23.22466498, 13.8985316 ,  5.04748584]])
    >>> arcma = smoothing (ar, view =True )# ma=True by default
    >>> arcma [:3, :3 ]
    array([[23.96547827,  8.48064226, 31.81490918],
           [26.21374675, 13.33233065, 12.29345026],
           [22.60143346, 16.77242118,  2.07931194]])
    >>> arcma_1 = smoothing (ar, view =True, axis =1 )
    >>> arcma_1 [:3, :3 ]
    array([[18.74017857, 26.91532187, 32.02914421],
           [18.4056216 , 21.81293014, 21.98535213],
           [-1.44359989,  3.49228057,  7.51734762]])
    """
    ar = np.array ( 
        is_iterable(ar, exclude_string = True , transform =True )
        ) 
    if ( 
            str (axis).lower().find('1')>=0 
            or str(axis).lower().find('column')>=0
            ): 
        axis = 1 
    else : axis =0 
    
    if _is_arraylike_1d(ar): 
        ar = reshape ( ar, axis = 0 ) 
    # make a copy
    # print(ar.shape )
    arr = ar.copy() 
    along_axis = arr.shape [1] if axis == 0 else len(ar) 
    arr0 = np.zeros_like (arr)
    for ix in range (along_axis): 
        value = arr [:, ix ] if axis ==0 else arr[ix , :]
        yc = smooth1d(value, drop_outliers = drop_outliers , 
                      ma= ma, view =False , absolute =absolute 
                      ) 
        if axis ==0: 
            arr0[:, ix ] = yc 
        else : arr0[ix, :] = yc 
        
    if view: 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                                 figsize = fig_size )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid', 
                     cmap = cmap )
        ax[1].imshow (arr0, interpolation ='nearest', label = 'Smooth Grid', 
                      cmap =cmap  )
        
        ax[0].set_title ('Raw Grid') 
        ax[0].set_xlabel (xlabel or '')
        ax[0].set_ylabel ( ylabel or '')
        ax[1].set_title ('Smooth Grid') 
        ax[1].set_xlabel (xlabel or '')
        ax[1].set_ylabel ( ylabel or '')
        
        plt.show () 
        
    if 1 in ar.shape: 
        arr0 = reshape (arr0 )
        
    return arr0 
    
def fittensor(
    refreq:ArrayLike , 
    compfreq: ArrayLike ,
    z: NDArray[DType[complex]] , 
    fill_value: Optional[float] = np.nan
)->NDArray[DType[complex]] : 
    """ Fit each tensor component to the complete frequency range. 
    
    The complete frequency is the frequency with clean data. It contain all the 
    frequency range on the site. During the survey, the missing frequencies 
    lead to missing tensor data. So the function will indicate where the tensor 
    data is missing and fit to the prior frequencies. 
    
    Parameters 
    ------------
    refreq: ArrayLike 
       Reference frequency - Should be the complete frequency collected 
       in the field. 
        
    comfreq: array-like, 
       The specific frequency collect in the site. Sometimes due to the 
       interferences, the frequency at individual site could be different 
       from the complete. However, the frequency values at the individual site 
       must be included in the complete frequency `refreq`. 
    
    z: array-like, 
       should be the  tensor value (real or imaginary part ) at 
       the component  xx, xy, yx, yy. 
        
    fill_value: float . default='NaN'
        Value to replace the missing data in tensors. 
        
    Returns
    -------
    Z: Arraylike 
       new Z filled by invalid value `NaN` where the frequency is missing 
       in the data. 

    Examples 
    ---------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import fittensor
    >>> refreq = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    >>> freq_ = np.hstack ((refreq.copy()[:7], refreq.copy()[12:] )) 
    >>> z = np.random.randn(len(freq_)) *10 # assume length of  freq as 
    ...                 # the same like the tensor Z value 
    >>> zn  = fittensor (refreq, freq_, z)
    >>> z # some frequency values are missing but not visible. 
    ...array([-23.23448367,   2.93185982,  10.81194723, -12.46326732,
             1.57312908,   7.23926576, -14.65645799,   9.85956253,
             3.96269863, -10.38325124,  -4.29739755,  -8.2591703 ,
            21.7930423 ,   0.21709129,   4.07815217])
    >>> # zn show where the frequencies are missing  
    >>> # the NaN value means in a missing value in  tensor Z at specific frequency  
    >>> zn 
    ... array([-23.23448367,   2.93185982,  10.81194723, -12.46326732,
             1.57312908,   7.23926576, -14.65645799,          nan,
                    nan,          nan,          nan,          nan,
             9.85956253,   3.96269863, -10.38325124,  -4.29739755,
            -8.2591703 ,  21.7930423 ,   0.21709129,   4.07815217])
    >>> # let visualize where the missing frequency value in tensor Z 
    >>> refreq 
    ... array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
           5.52631581e+07, 5.15789476e+07, 4.78947372e+07, 4.42105267e+07*,
           4.05263162e+07*, 3.68421057e+07*, 3.31578953e+07*, 2.94736848e+07*,
           2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
           1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    >>> refreq[np.isnan(zn)] #we can see the missing value between [7:12](*) in refreq 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368])
    
    """
    refreq = check_y (refreq, input_name="Reference array 'refreq'")
    freqn, mask = ismissing(refarr= refreq , arr =compfreq, 
                            return_index='mask',fill_value = fill_value
                            )
    #mask_isin = np.isin(refreq, compfreq)
    z_new = np.full_like(freqn, fill_value = fill_value, 
                         dtype = z.dtype 
                         ) 

    if len(z_new[mask]) != len(reshape(z) ): 
        raise EMError (
            "Fitting tensor cannot be performed with inconsistent frequencies."
            " Frequency in Z must be consistent for all investigated sites,"
            " i.e. the frequencies values in Z must be included in the complete"
            f" frequency array (`refreq`) for all sites. Got {len(z_new[mask])}"
            f" while expecting {len(reshape(z))}. If frequencies are inputted"
            " manually, use `hlearn.utils.exmath.find_closest` to get the closest"
            " frequencies from the input ones. "
            )
    z_new[mask] = reshape(z) 
    
    return z_new 
    
def interpolate1d (
        arr:ArrayLike[DType[T]], 
        kind:str = 'slinear', 
        method:str=None, 
        order:Optional[int] = None, 
        fill_value:str ='extrapolate',
        limit:Tuple[float] =None, 
        **kws
    )-> ArrayLike[DType[T]]:
    """ Interpolate array containing invalid values `NaN`
    
    Usefull function to interpolate the missing frequency values in the 
    tensor components. 
    
    Parameters 
    ----------
    arr: array_like 
        Array to interpolate containg invalid values. The invalid value here 
        is `NaN`. 
        
    kind: str or int, optional
        Specifies the kind of interpolation as a string or as an integer 
        specifying the order of the spline interpolator to use. The string 
        has to be one of ``linear``, ``nearest``, ``nearest-up``, ``zero``, 
        ``slinear``,``quadratic``, ``cubic``, ``previous``, or ``next``. 
        ``zero``, ``slinear``, ``quadratic``and ``cubic`` refer to a spline 
        interpolation of zeroth, first, second or third order; ``previous`` 
        and ``next`` simply return the previous or next value of the point; 
        ``nearest-up`` and ``nearest`` differ when interpolating half-integers 
        (e.g. 0.5, 1.5) in that ``nearest-up`` rounds up and ``nearest`` rounds 
        down. If `method` param is set to ``pd`` which refers to pd.interpolate 
        method , `kind` can be set to ``polynomial`` or ``pad`` interpolation. 
        Note that the polynomial requires you to specify an `order` while 
        ``pad`` requires to specify the `limit`. Default is ``slinear``.
        
    method: str, optional, default='mean' 
        Method of interpolation. Can be ``base`` for `scipy.interpolate.interp1d`
        ``mean`` or ``bff`` for scaling methods and ``pd``for pandas interpolation 
        methods. Note that the first method is fast and efficient when the number 
        of NaN in the array if relatively few. It is less accurate to use the 
        `base` interpolation when the data is composed of many missing values.
        Alternatively, the scaled method(the  second one) is proposed to be the 
        alternative way more efficient. Indeed, when ``mean`` argument is set, 
        function replaces the NaN values by the nonzeros in the raw array and 
        then uses the mean to fit the data. The result of fitting creates a smooth 
        curve where the index of each NaN in the raw array is replaced by its 
        corresponding values in the fit results. The same approach is used for
        ``bff`` method. Conversely, rather than averaging the nonzeros values, 
        it uses the backward and forward strategy  to fill the NaN before scaling.
        ``mean`` and ``bff`` are more efficient when the data are composed of 
        lot of missing values. When the interpolation `method` is set to `pd`, 
        function uses the pandas interpolation but ended the interpolation with 
        forward/backward NaN filling since the interpolation with pandas does
        not deal with all NaN at the begining or at the end of the array. Default 
        is ``base``.
        
    fill_value: array-like or (array-like, array_like) or ``extrapolate``, optional
        If a ndarray (or float), this value will be used to fill in for requested
        points outside of the data range. If not provided, then the default is
        NaN. The array-like must broadcast properly to the dimensions of the 
        non-interpolation axes.
        If a two-element tuple, then the first element is used as a fill value
        for x_new < x[0] and the second element is used for x_new > x[-1]. 
        Anything that is not a 2-element tuple (e.g., list or ndarray,
        regardless of shape) is taken to be a single array-like argument meant 
        to be used for both bounds as below, above = fill_value, fill_value.
        Using a two-element tuple or ndarray requires bounds_error=False.
        Default is ``extrapolate``. 
        
    kws: dict 
        Additional keyword arguments from :class:`spi.interp1d`. 
    
    Returns 
    -------
    array like - New interpoolated array. `NaN` values are interpolated. 
    
    Notes 
    ----- 
    When interpolated thoughout the complete frequencies  i.e all the frequency 
    values using the ``base`` method, the missing data in `arr`  can be out of 
    the `arr` range. So, for consistency and keep all values into the range of 
    frequency, the better idea is to set the param `fill_value` in kws argument
    of ``spi.interp1d`` to `extrapolate`. This will avoid an error to raise when 
    the value to  interpolated is extra-bound of `arr`. 
    
    
    References 
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    https://www.askpython.com/python/examples/interpolation-to-fill-missing-entries
    
    Examples 
    --------
    >>> import numpy as np 
    >>> import matplotlib.pyplot as plt 
    >>> from hlearn.utils.exmath  import interpolate1d,
    >>> z = np.random.randn(17) *10 # assume 17 freq for 17 values of tensor Z 
    >>> z [[7, 10, 16]] =np.nan # replace some indexes by NaN values 
    >>> zit = interpolate1d (z, kind ='linear')
    >>> z 
    ... array([ -1.97732415, -16.5883156 ,   8.44484348,   0.24032979,
              8.30863276,   4.76437029, -15.45780568,          nan,
             -4.11301794, -10.94003412,          nan,   9.22228383,
            -15.40298253,  -7.24575491,  -7.15149205, -20.9592011 ,
                     nan]),
    >>> zn 
    ...array([ -1.97732415, -16.5883156 ,   8.44484348,   0.24032979,
             8.30863276,   4.76437029, -15.45780568,  -4.11301794,
           -10.94003412,   9.22228383, -15.40298253,  -7.24575491,
            -7.15149205, -20.9592011 , -34.76691014, -48.57461918,
           -62.38232823])
    >>> zmean = interpolate1d (z,  method ='mean')
    >>> zbff = interpolate1d (z, method ='bff')
    >>> zpd = interpolate1d (z,  method ='pd')
    >>> plt.plot( np.arange (len(z)),  zit, 'v--', 
              np.arange (len(z)), zmean, 'ok-',
              np.arange (len(z)), zbff, '^g:',
              np.arange (len(z)), zpd,'<b:', 
              np.arange (len(z)), z,'o', 
              )
    >>> plt.legend(['interp1d', 'mean strategy', 'bff strategy',
                    'pandas strategy', 'data'], loc='best')
    
    """
    method = method or 'mean'; method =str(method).strip().lower() 
    if method in ('pandas', 'pd', 'series', 'dataframe','df'): 
        method = 'pd' 
    elif method in ('interp1d', 'scipy', 'base', 'simpler', 'i1d'): 
        method ='base' 
    
    if not hasattr (arr, '__complex__'): 
        
        arr = check_y(arr, allow_nan= True, to_frame= True ) 
    # check whether there is nan and masked invalid 
    # and take only the valid values 
    t_arr = arr.copy() 
    
    if method =='base':
        mask = ~np.ma.masked_invalid(arr).mask  
        arr = arr[mask] # keep the valid values
        f = spi.interp1d( x= np.arange(len(arr)), y= arr, kind =kind, 
                         fill_value =fill_value, **kws) 
        arr_new = f(np.arange(len(t_arr)))
        
    if method in ('mean', 'bff'): 
        arr_new = arr.copy()
        
        if method =='mean': 
            # use the mean of the valid value
            # and fill the nan value
            mean = t_arr[~np.isnan(t_arr)].mean()  
            t_arr[np.isnan(t_arr)]= mean  
            
        if method =='bff':
            # fill NaN values back and forward.
            t_arr = fillNaN(t_arr, method = method)
            t_arr= reshape(t_arr)
            
        yc, *_= scaley (t_arr)
        # replace the at NaN positions value in  t_arr 
        # with their corresponding scaled values 
        arr_new [np.isnan(arr_new)]= yc[np.isnan(arr_new)]
        
    if method =='pd': 
        t_arr= pd.Series (t_arr, dtype = t_arr.dtype )
        t_arr = np.array(t_arr.interpolate(
            method =kind, order=order, limit = limit ))
        arr_new = reshape(fillNaN(t_arr, method= 'bff')) # for consistency 
        
    return arr_new 
   

def moving_average (
    y:ArrayLike[DType[T]],
    *, 
    window_size:int  = 3 , 
    method:str  ='sma',
    mode:str  ='same', 
    alpha: int  =.5 
)-> ArrayLike[DType[T]]: 
    """ A moving average is  used with time series data to smooth out
    short-term fluctuations and highlight longer-term trends or cycles.
    
    Funtion analyzes data points by creating a series of averages of different
    subsets of the full data set. 
    
    Parameters 
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
        
    window_size : int
        the length of the window. Must be greater than 1 and preferably
        an odd integer number.Default is ``3``
        
    method: str 
        variant of moving-average. Can be ``sma``, ``cma``, ``wma`` and ``ema`` 
        for simple, cummulative, weight and exponential moving average. Default 
        is ``sma``. 
        
    mode: str
        returns the convolution at each point of overlap, with an output shape
        of (N+M-1,). At the end-points of the convolution, the signals do not 
        overlap completely, and boundary effects may be seen. Can be ``full``,
        ``same`` and ``valid``. See :doc:`~np.convole` for more details. Default 
        is ``same``. 
        
    alpha: float, 
        smoothing factor. Only uses in exponential moving-average. Default is 
        ``.5``.
    
    Returns 
    --------
    ya: array like, shape (N,) 
        Averaged time history of the signal
    
    Notes 
    -------
    The first element of the moving average is obtained by taking the average 
    of the initial fixed subset of the number series. Then the subset is
    modified by "shifting forward"; that is, excluding the first number of the
    series and including the next value in the subset.
    
    Examples
    --------- 
    >>> import numpy as np ; import matplotlib.pyplot as plt 
    >>> from hlearn.utils.exmath  import moving_average 
    >>> data = np.random.randn (37) 
    >>> # add gaussion noise to the data 
    >>> data = 2 * np.sin( data)  + np.random.normal (0, 1 , len(data))
    >>> window = 5  # fixed size to 5 
    >>> sma = moving_average(data, window) 
    >>> cma = moving_average(data, window, method ='cma' )
    >>> wma = moving_average(data, window, method ='wma' )
    >>> ema = moving_average(data, window, method ='ema' , alpha =0.6)
    >>> x = np.arange(len(data))
    >>> plt.plot (x, data, 'o', x, sma , 'ok--', x, cma, 'g-.', x, wma, 'b:')
    >>> plt.legend (['data', 'sma', 'cma', 'wma'])
    
    References 
    ----------
    .. * [1] https://en.wikipedia.org/wiki/Moving_average
    .. * [2] https://www.sciencedirect.com/topics/engineering/hanning-window
    .. * [3] https://stackoverflow.com/questions/12816011/weighted-moving-average-with-numpy-convolve
    
    """
    y = np.array(y)
    try:
        window_size = np.abs(_assert_all_types(int(window_size), int))
    except ValueError:
        raise ValueError("window_size has to be of type int")
    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if  window_size > len(y):
        raise TypeError("window_size is too large for averaging. Window"
                        f" must be greater than 0 and less than {len(y)}")
    
    method =str(method).lower().strip().replace ('-', ' ') 
    
    if method in ('simple moving average',
                  'simple', 'sma'): 
        method = 'sma' 
    elif method  in ('cumulative average', 
                     'cumulative', 'cma'): 
        method ='cma' 
    elif method  in ('weighted moving average',
                     'weight', 'wma'): 
        method = 'wma'
    elif method in('exponential moving average',
                   'exponential', 'ema'):
        method = 'ema'
    else : 
        raise ValueError ("Variant average methods only includes "
                          f" {smart_format(['sma', 'cma', 'wma', 'ema'], 'or')}")
    if  1. <= alpha <= 0 : 
        raise ValueError ('alpha should be less than 1. and greater than 0. ')
        
    if method =='sma': 
        ya = np.convolve(y , np.ones (window_size), mode ) / window_size 
        
    if method =='cma': 
        y = np.cumsum (y) 
        ya = np.array([ y[ii]/ len(y[:ii +1]) for ii in range(len(y))]) 
        
    if method =='wma': 
        w = np.cumsum(np.ones(window_size, dtype = float))
        w /= np.sum(w)
        ya = np.convolve(y, w[::-1], mode ) #/window_size
        
    if method =='ema': 
        ya = np.array ([y[0]]) 
        for ii in range(1, len(y)): 
            v = y[ii] * alpha + ( 1- alpha ) * ya[-1]
            ya = np.append(ya, v)
            
    return ya 


def get_profile_angle (
        easting: float =None, northing: float =None, msg:str ="ignore" ): 
    """
    compute geoprofile angle. 
    Parameters 
    -----------
    * easting : array_like 
            easting coordiantes values 
    * northing : array_like 
            northing coordinates values
    * msg: output a little message if msg is set to "raises"
    
    Returns 
    ---------
    float
         profile_angle 
    float 
        geo_electric_strike 
    """
    msg = (
        "Need to import scipy.stats as a single module. Sometimes import scipy "
        "differently  with stats may not work. Use either `import scipy.stats`"
        " rather than `import scipy as sp`" 
        )
    
    if easting is None or northing is None : 
        raise TypeError('NoneType can not be computed !')
        
        # use the one with the lower standard deviation
    try :
        easting = easting.astype('float')
        northing = northing.astype('float')
    except : 
        raise ValueError('Could not convert input argument to float!')
    try : 
        profile1 = spstats.linregress(easting, northing)
        profile2 =spstats.linregress(northing, easting)
    except:
        warnings.warn(msg)
        
    profile_line = profile1[:2]
    # if the profile is rather E=E(N),
    # the parameters have to converted  into N=N(E) form:
    
    if profile2[4] < profile1[4]:
        profile_line = (1. / profile2[0], -profile2[1] / profile2[0])

    # if self.profile_angle is None:
    profile_angle = (90 - (np.arctan(profile_line[0]) * 180 / np.pi)) % 180
    
    # otherwise: # have 90 degree ambiguity in 
    #strike determination# choose strike which offers larger
    #  angle with profile if profile azimuth is in [0,90].
    if msg=="raises": 
        print("+++ -> Profile angle is {0:+.2f} degrees E of N".format(
                profile_angle
                ) )
    return np.around( profile_angle,2)
     
def get_strike (
        profile_angle:float = None, 
        easting =None, northing:float=None, 
        gstrike:float =None, 
        msg:str="ignore"
        )->Tuple[float, float, str]:
    """
    Compute geoelectric strike from profile angle, easting and northing.
    
    Parameters
    -------------
    *  profile_angle : float 
        If not provided , will comput with easting and northing coordinates 
    * easting : array_like 
        Easting coordiantes values 
    * northing : array_like 
        Northing coordinates values 
    * gstrike : float 
        strike value , if provided, will recomputed geo_electric strike .
     * msg: output a little message if msg is set to "raises"
     
    Returns 
    --------
    float
         profile_angle in degree E of N 
    float 
        geo_electric_strike in degrees E of N
     
    """
    
    if profile_angle is None and  easting is None and northing is None : 
        _logger.debug("NoneType is given. Use 'gstrike' to recompute the "
                      "geoelectrical strike")
        if gstrike is None :
            raise TypeError("Could not compute geo-electrike strike!")
    
    if profile_angle is None : 
        if easting is not None and northing is not None : 
            profile_angle ,_ = get_profile_angle(
                                easting, northing)
    
    if gstrike is None : 
        if 0<= profile_angle < 90 :
            geo_electric_strike  = profile_angle + 90  
        elif 90<=profile_angle < 180 :
            geo_electric_strike = profile_angle -90
        elif 180 <= profile_angle <270 :
            geo_electric_strike = - profile_angle +90 
        else :
            geo_electric_strike  = - profile_angle -90 
        
        geo_electric_strike  %= 180   
    
    if gstrike is not None : # recomputed geo_electrike strike 
        if 0 <= profile_angle < 90:
            if np.abs(profile_angle - gstrike) < 45:
                geo_electric_strike  = gstrike+ 90
     
        elif 90 <= profile_angle < 135:
            if profile_angle - gstrike < 45:
                geo_electric_strike = gstrike - 90
        else:
            if profile_angle - gstrike >= 135:
               geo_electric_strike = gstrike+ 90
        geo_electric_strike %=  180         # keep value of
        #geoelectrike strike less than 180 degree
        
    geo_electric_strike =np.floor(geo_electric_strike)
    if msg=="raises": 
        print("+++ -> Profile angle is {0:+.2f} degrees E of N".format(
            geo_electric_strike))
    return  geo_electric_strike, profile_angle 
        
        

def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None,
                  use="conv"):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be an odd positive integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.

    See Also
    --------
    savgol_filter

    Examples
    --------
    >>> from hlearn.exmath.signal import savgol_coeffs
    >>> savgol_coeffs(5, 2)
    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
    >>> savgol_coeffs(5, 2, deriv=1)
    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,
           -2.00000000e-01])

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.

    >>> x = np.array([1, 0, 1, 4, 9])
    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    >>> c.dot(x)
    6.0
    """

    # An alternative method for finding the coefficients when deriv=0 is
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # The method implemented here is faster.

    # To recreate the table of sample coefficients shown in the chapter on
    # the Savitzy-Golay filter in the Numerical Recipes book, use
    #    window_length = nL + nR + 1
    #    pos = nL + 1
    #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if rem == 0:
        raise ValueError("window_length must be odd.")

    if pos is None:
        pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")

    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)
    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    # The coefficient assigned to y[deriv] scales the result to take into
    # account the order of the derivative and the sample spacing.
    y[deriv] = float_factorial(deriv) / (delta ** deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs


def _polyder(p, m):
    """Differentiate polynomials represented with coefficients.

    p must be a 1-D or 2-D array.  In the 2-D case, each column gives
    the coefficients of a polynomial; the first row holds the coefficients
    associated with the highest power. m must be a nonnegative integer.
    (numpy.polyder doesn't handle the 2-D case.)
    """

    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = np.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = np.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


def get2dtensor(
    z_or_edis_obj_list:List[EDIO |ZO], /, 
    tensor:str= 'z', 
    component:str='xy', 
    kind:str ='modulus',
    return_freqs:bool=False, 
    **kws 
    ): 
    """ Make  tensor into two dimensional array from a 
    collection of Impedance tensors Z.
    
    Out 2D resistivity, phase-error and tensor matrix from a collection
    of EDI-objects. 
    
    Matrix depends of the number of frequency times number of sites. 
    The function asserts whether all data from all frequencies are available. 
    The missing values should be filled by NaN. Note that each element 
    of z is (nfreq, 2, 2) dimension for:
    
    .. code-block:: default 
       
       xx ( 0, 0) ------- xy ( 0, 1)
       yx ( 1, 0) ------- yy ( 1, 1) 
       
    Parameters 
    ----------- 

    z_or_edis_obj_list: list of :class:`hlearn.edi.Edi` or \
        :class:`hlearn.externals.z.Z` 
        A collection of EDI- or Impedances tensors objects. 
    
    tensor: str, default='z'  
        Tensor name. Can be [ resistivity|phase|z|frequency]
        
    component: str, default='xy' (TE mode)
        EM mode. Can be ['xx', 'xy', 'yx', 'yy']
      
    out: str 
        kind of data to output. Be sure to provide the component to retrieve 
        the attribute from the collection object. Except the `error` and 
        frequency attribute, the missing component to the attribute will 
        raise an error. for instance ``resxy`` for xy component. Default is 
        ``resxy``. 
        
    kind: str , default='modulus'
        focuses on the tensor output. Note that the tensor is a complex number 
        of ndarray (nfreq, 2,2 ). If set to``modulus`, the modulus of the complex 
        tensor should be outputted. If ``real`` or``imag``, it returns only
        the specific one. Default is ``complex``.

    return_freqs: Arraylike , 
        If ``True`` , returns also the full frequency ranges. 
    kws: dict 
        Additional keywords arguments from :meth:`~EM.getfullfrequency `. 
    
    Returns 
    -------- 
    mat2d: arraylike2d
        the matrix of number of frequency and number of Edi-collectes which 
        correspond to the number of the stations/sites. 
    
    Examples 
    ---------
    >>> from hlearn.datasets import load_huayuan
    >>> from hlearn.methods import get2dtensor 
    >>> box= load_huayuan ( key ='raw', clear_cache = True, samples =7)
    >>> data = box.data 
    >>> phase_yx = get2dtensor ( data, tensor ='phase', component ='yx')
    >>> phase_yx.shape 
    (56, 7)
    >>> phase_yx [0, :]
    array([        nan,         nan,         nan,         nan, 18.73244951,
           35.00516522, 59.91093054])
    """

    name, m2 = _validate_tensor (tensor = tensor, component = component, **kws)
    if name =='_freq': 
        raise EMError ("Tensor from 'Frequency' is not allowed here."
                       " Use `make2d` method instead: 'hlearn.EM.make2d'")
    if z_or_edis_obj_list is None: 
        raise EMError(f"Cannot output {name!r} 2D block with missing a"
                      " collection of EDI or Z objects.")
    # assert z and Edi objets 
    obj_type  = _assert_z_or_edi_objs (z_or_edis_obj_list)
    # get the frequency 
    freqs = get_full_frequency(z_or_edis_obj_list)
    # freqs = ( z_or_edis_obj_list[0].Z.freq if obj_type =='EDI'
    #          else z_or_edis_obj_list[0].freq ) 
    
    _c= {
          'xx': [slice (None, len(freqs)), 0 , 0] , 
          'xy': [slice (None, len(freqs)), 0 , 1], 
          'yx': [slice (None, len(freqs)), 1 , 0], 
          'yy': [slice (None, len(freqs)), 1,  1] 
    }

    zl = [getattr( ediObj.Z if obj_type =='EDI' else ediObj,
                  f"{name}")[tuple (_c.get(m2))]
          for ediObj in z_or_edis_obj_list ]

    try : 
        mat2d = np.vstack (zl ).T 
    except: 
        zl = [fittensor(freqs, ediObj.Z._freq 
                        if obj_type =='EDI' else ediObj.freq , v)
              for ediObj ,  v  in zip(z_or_edis_obj_list, zl)]
        # stacked the z values alomx axis=1. 
        # return np.hstack ([ reshape (o, axis=0) for o in zl])
        mat2d = concat_array_from_list (zl , concat_axis=1) 
        
    if 'z' in name: 
        zdict = {'modulus': np.abs (mat2d), 'real': mat2d.real, 
         'imag': mat2d.imag, 'complex':mat2d
         } 
    
        mat2d = zdict [kind]
        
    return mat2d if not return_freqs else (mat2d, freqs  )

def get_full_frequency (
        z_or_edis_obj_list: List [EDIO |ZO], 
        /,
        to_log10:bool  =False 
    )-> ArrayLike[DType[float]]: 
    """ Get the frequency with clean data. 
    
    The full or plain frequency is array frequency with no missing frequency
    during the data collection. Note that when using |NSAMT|, some data 
    are missing due to the weak of missing frequency at certain band 
    especially in the attenuation band. 

    Parameters 
    -----------
    z_or_edis_obj_list: list of :class:`hlearn.edi.Edi` or \
        :class:`hlearn.externals.z.Z` 
        A collection of EDI- or Impedances tensors objects. 
        
    to_log10: bool, default=False 
       Export frequency to base 10 logarithm 
       
    Returns 
    -------
    f : Arraylike of shape(N, )
       frequency with clean data. Out of `attenuation band` if survey 
       is completed with  |NSAMT|. 
    
    Examples 
    --------
    >>> from hlearn.datasets import load_huayuan
    >>> from hlearn.methods.em import get_full_frequency
    >>> box= load_huayuan ( key ='raw', clear_cache = True, samples =7)
    >>> edi_data = box.data
    >>> f = get_full_frequency (edi_data )
    >>> f 
    array([8.19200e+04, 7.00000e+04, 5.88000e+04, 4.95000e+04, 4.16000e+04,
           3.50000e+04, 2.94000e+04, 2.47000e+04, 2.08000e+04, 1.75000e+04,
           ...
           3.25000e+01, 2.75000e+01, 2.25000e+01, 1.87500e+01, 1.62500e+01,
           1.37500e+01, 1.12500e+01, 9.37500e+00, 8.12500e+00, 6.87500e+00,
           5.62500e+00])
    >>> len(f) 
    56
    >>> # Get only the z component objects 
    >>> zobjs = [ box.emo.ediObjs_[i].Z for i in  range (len(box.emo.ediObjs_))]
    >>> len(zobjs)
    56 
    """
    obj_type  = _assert_z_or_edi_objs (z_or_edis_obj_list)
    
    lenfs = np.array([len(ediObj.Z._freq if obj_type =='EDI' else ediObj.freq )
                      for ediObj in z_or_edis_obj_list ] ) 
    ix_fm = np.argmax (lenfs) 
    f=  ( z_or_edis_obj_list [ix_fm].Z._freq if obj_type =='EDI' 
         else z_or_edis_obj_list [ix_fm]._freq 
         ) 
    return np.log10(f) if to_log10 else f 
    
#XXX OPTIMIZE 
def compute_errors (
        arr, /, 
        error ='std', 
        axis = 0, 
        return_confidence=False 
        ): 
    """ Compute Errors ( Standard Deviation ) and standard errors. 
    
    Standard error and standard deviation are both measures of variability:
    - The standard deviation describes variability within a single sample. Its
      formula is given as: 
          
      .. math:: 
          
          SD = \sqrt{ \sum |x -\mu|^2}{N}
          
      where :math:`\sum` means the "sum of", :math:`x` is the value in the data 
      set,:math:`\mu` is the mean of the data set and :math:`N` is the number 
      of the data points in the population. :math:`SD` is the quantity 
      expressing by how much the members of a group differ from the mean 
      value for the group.
      
    - The standard error estimates the variability across multiple 
      samples of a population. Different formulas are used depending on 
      whether the population standard deviation is known.
      
      - when the population standard deviation is known: 
      
        .. math:: 
          
            SE = \frac{SD}{\sqrt{N}} 
            
      - When the population parameter is unknwon 
      
        .. math:: 
            
            SE = \frac{s}{\sqrt{N}} 
            
       where :math:`SE` is the standard error, : math:`s` is the sample
       standard deviation. When the population standard is knwon the 
       :math:`SE` is more accurate. 
    
    Note that the :math:`SD` is  a descriptive statistic that can be 
    calculated from sample data. In contrast, the standard error is an 
    inferential statistic that can only be estimated 
    (unless the real population parameter is known). 
    
    Parameters
    ----------
    arr : array_like , 1D or 2D 
      Array for computing the standard deviation 
      
    error: str, default='std'
      Name of error to compute. By default compute the standard deviation. 
      Can also compute the the standard error estimation if the  argument 
      is passed to ``ste``. 
    return_confidence: bool, default=False, 
      If ``True``, returns the confidence interval with 95% of sample means 
      
    Returns 
    --------
    err: arraylike 1D or 2D 
       Error array. 
       
    Examples
    ---------
    >>> from hlearn.datasets import load_huayuan 
    >>> from hlearn.utils.exmath import compute_errors
    >>> emobj=load_huayuan ().emo
    >>> compute_errors (emobj.freqs_ ) 
    .. Out[104]: 14397.794665683341
    >>> freq2d = emobj.make2d ('freq') 
    >>> compute_errors (freq2d ) [:7]
    array([14397.79466568, 14397.79466568, 14397.79466568, 14397.79466568,
           14397.79466568, 14397.79466568, 14397.79466568])
    >>> compute_errors (freq2d , error ='se') [:7]
    array([1959.29168624, 1959.29168624, 1959.29168624, 1959.29168624,
           1959.29168624, 1959.29168624, 1959.29168624])
    
    """
    error = _validate_name_in(error , defaults =('error', 'se'),
                              deep =True, expect_name ='se')

    err= np.std (arr) if arr.ndim ==1 else np.std (arr, axis= axis )
                  
    err_lower =  err_upper = None 
    if error =='se': 
        N = len(arr) if arr.ndim ==1 else arr.shape [axis ]
        err =  err / np.sqrt(N)
        if return_confidence: 
            err_lower = arr.mean() - ( 1.96 * err ) 
            err_upper = arr.mean() + ( 1.96 * err )
    return err if not return_confidence else ( err_lower, err_upper)  

def plot_confidence_in(
    z_or_edis_obj_list: List [EDIO |ZO], 
    /, 
    tensor:str='res', 
    view:str='1d', 
    drop_outliers:bool=True, 
    distance:float=None, 
    c_line:bool =False,
    view_ci:bool=True, 
    figsize:Tuple=(6, 2), 
    fontsize:bool=4., 
    dpi:int=300., 
    top_label:str='Stations',
    rotate_xlabel:float=90., 
    fbtw:bool =True, 
    savefig: str=None, 
    **plot_kws
    ): 
    """Plot data confidency from tensor errors. 
    
    The default :term:`tensor` for evaluating the data confidence is the resistivity 
    at TE mode ('xy'). 
    
    Check confidence in the data before starting the concrete processing 
    seems meaningful. In the area with complex terrain, with high topography 
    addition to interference noises, signals are weals or missing 
    especially when using :term:`AMT` survey. The most common technique to 
    do this is to eliminate the bad frequency and interpolate the remains one. 
    However, the tricks for eliminating frequency differ from one author 
    to another. Here, the tip using the data confidence seems meaningful
    to indicate which frequencies to eliminate (at which stations/sites)
    and which ones are still recoverable using the tensor recovering 
    strategy. 
    
    The plot implements three levels of confidence: 
        
    - High confidence: :math:`conf. \geq 0.95` values greater than 95% 
    - Soft confidence: :math:`0.5 \leq conf. < 0.95`. The data in this 
      confidence range can be beneficial for tensor recovery to restore 
      the weak and missing signals. 
    - bad confidence: :math:`conf. <0.5`. Data in this interval must be 
      deleted.

    Parameters 
    -----------
    z_or_edis_obj_list: list of :class:`hlearn.edi.Edi` or \
        :class:`hlearn.externals.z.Z` 
        A collection of EDI- or Impedances tensors objects. 
        
    tensor: str, default='res'  
        Tensor name. Can be [ 'resistivity'|'phase'|'z'|'frequency']
        
    view:str, default='1d'
       Type of plot. Can be ['1D'|'2D'] 
       
    drop_outliers: bool, default=True 
       Suppress the ouliers in the data if ``True``. 
       
    distance: float, optional 
       Distance between stations/sites 
       
    fontsize: float,  default=3. 
       label font size. 
    
    figsize: Tuple, default=(6, 2)
       Figure size. 
       
    c_line: bool, default=True, 
       Display the confidence line in two dimensinal view.  
       
    dpi: int, default=300 
       Image resolution in dot-per-inch 
       
    rotate_xlabel: float, default=90.
       Angle to rotate the stations/sites labels 
       
    top_labels: str,default='Stations' 
       Labels the sites either using the survey name. 
       
    view_ci: bool,default=True, 
       Show the marker of confidence interval. 
       
    fbtw: bool, default=True, 
       Fill between confidence interval. 
       
    plot_kws: dict, 
       Additional keywords pass to the :func:`~mplt.plot`
       
    See Also
    ---------
    hlearn.methods.Processing.zrestore: 
        For more details about the function for tensor recovering technique. 
        
    Examples 
    ----------
    >>> from hlearn.utils.exmath import plot_confidence_in 
    >>> from hlearn.datasets import fetch_data 
    >>> emobj  = fetch_data ( 'huayuan', samples = 25, clear_cache =True,
                             key='raw').emo
    >>> plot_confidence_in (emobj.ediObjs_ , 
                            distance =20 , 
                            view ='2d', 
                            figsize =(6, 2)
                            )
    >>> plot_confidence_in (emobj.ediObjs_ , distance =20 ,
                            view ='1d', figsize =(6, 3), fontsize =5, 
                            )
    """
    from .plotutils import _get_xticks_formatage 
    
    # by default , we used the resistivity tensor and error at TE mode.
    # force using the error when resistivity or phase tensors are supplied 
    tensor = str(tensor).lower() ; view = str(view).lower() 
    tensor = tensor + '_err' if tensor in 'resistivityphase' else tensor 
    rerr, freqs = get2dtensor(z_or_edis_obj_list, tensor =tensor,
                                return_freqs=True )
    ratio_0 = get_confidence_ratio(rerr ) # alongside columns (stations )
    #ratio_1 = get_confidence_ratio(rerr , axis =1 ) # along freq 
    # make confidencity properties ( index, colors, labels ) 
    conf_props = dict (# -- Good confidencity 
                       high_cf = (np.where ( ratio_0 >= .95  )[0] ,  
                                   '#15B01A','$conf. \geq 0.95$' ), 
                       # -- might be improve using tensor recovering 
                       soft_cf = (np.where ((ratio_0 < .95 ) &(ratio_0 >=.5 ))[0], 
                                  '#FF81C0', '$0.5 \leq conf. < 0.95$'), 
                       # --may be deleted 
                       bad_cf= (np.where ( ratio_0 < .5 )[0], 
                                '#650021','$conf. <0.5$' )
                       )
    # re-compute distance 
    distance = distance or 1. 
    d= np.arange ( rerr.shape[1])  * distance 
    # format clabel for error 
    clab=r"resistivity ($\Omega$.m)" if 'res' in tensor else (
        r'phase ($\degree$)' if 'ph' in tensor else tensor )
    # --plot 
    if view =='2d': 
        from ..view import plot2d
        ar2d = remove_outliers(rerr, fill_value=np.nan
                              ) if drop_outliers else rerr 
       
        ax = plot2d (
              ar2d,
              cmap ='binary', 
              cb_label =f"Error in {clab}", 
              top_label =top_label , 
              rotate_xlabel = rotate_xlabel , 
              distance = distance , 
              y = np.log10 (freqs), 
              fig_size  = figsize ,
              fig_dpi = dpi , 
              font_size =fontsize,
              )
        
    else: 
        fig, ax = plt.subplots(figsize = figsize,  dpi = dpi ,
                               )
        ax.plot(d , ratio_0  , 'ok-', markersize=2.,  #alpha = 0.5,
                **plot_kws)
        if fbtw:
            # use the minum to extend the plot line 
            min_sf_ci = .5 if ratio_0.min() <=0.5 else ratio_0.min() 
            # -- confidence condition 
            ydict =dict(yh =np.repeat(.95  , len(ratio_0)), 
                        sh = np.repeat( min_sf_ci , len(ratio_0 ))
                        )
            rr= ( ratio_0 >=0.95 , (ratio_0 < .95 ) & (ratio_0 >=min_sf_ci ), 
                 ratio_0 < min_sf_ci )
            
            for ii, rat in enumerate (rr): 
                if len(rat)==0: break 
                ax.fill_between(d, ratio_0, 
                                ydict ['sh'] if ii!=0 else ydict ['yh'],
                                facecolor = list( conf_props.values())[ii][1], 
                                where = rat, 
                                alpha = .3 , 
                                )
                ax.axhline(y=min_sf_ci if ii!=0 else .95, 
                            color="k",
                            linestyle="--", 
                            lw=1. 
                            )
                
        ax.set_xlabel ('Distance (m)', fontsize =1.2 * fontsize,
                       fontdict ={'weight': 'bold'})
        ax.set_ylabel ("Confidence ratio x100 (%)", fontsize = 1.2 * fontsize , 
                       fontdict ={'weight': 'bold'}
                       )
        ax.tick_params (labelsize = fontsize)
        ax.set_xlim ([ d.min(), d.max() ])
        
        # make twin axis to upload the stations 
        #--> set second axis 
        axe2 = ax.twiny() 
        axe2.set_xticks(range(len(d)),minor=False )
        
        # set ticks params to reformat the size 
        axe2.tick_params (  labelsize = fontsize)
        # get xticks and format labels using the auto detection 
    
        _get_xticks_formatage(axe2, range(len(d)), fmt = 'E{:02}',  
                              auto=True, 
                              rotation=rotate_xlabel 
                              )
        
        axe2.set_xlabel(top_label, fontdict ={
            'size': fontsize ,
            'weight': 'bold'}, )
        
    #--plot confidency 
    if view_ci: 
        if view=='2d' and c_line: 
           # get default line properties 
           c= plot_kws.pop ('c', 'r') 
           lw = plot_kws.pop ('lw', .5)
           ls = plot_kws.pop ('ls', '-')
           
           ax.plot (d, ratio_0 *np.log10 (freqs).max() , 
                    ls=ls, 
                    c=c , 
                    lw=lw, 
                    label='Confidence line'
                    )
        
        for cfv, c , lab in conf_props.values (): 
            if len(cfv)==0: break 
            norm_coef  =  np.log10 (freqs).max() if view =='2d' else 1. 
            ax.scatter (d[cfv], ratio_0[cfv] * norm_coef,
                          marker ='o', 
                          edgecolors='k', 
                          color= c,
                          label = lab, 
                          )
            ax.legend(loc ='lower right' if view=='2d' else 'best') 

    if savefig: 
        plt.savefig(savefig, dpi =600 )
        
    # plot when image is saved and show otherwise 
    plt.show() if not savefig else plt.close() 
        
    return ax 


def get_z_from( edi_obj_list , /, ): 
    """Extract z object from Edi object.
    
    Parameters 
    -----------
    z_or_edis_obj_list: list of :class:`hlearn.edi.Edi` or \
        :class:`hlearn.externals.z.Z` 
        A collection of EDI- or Impedances tensors objects. 
    Returns
    --------
    Z: list of :class:`hlearn.externals.z.Z`
       List of impedance tensor Objects. 
      
    """
    obj_type  = _assert_z_or_edi_objs (edi_obj_list)
    return   edi_obj_list  if obj_type =='z' else [
        edi_obj_list[i].Z for i in range (len( edi_obj_list)  )] 

def qc(
    z_or_edis_obj_list: List [EDIO |ZO], 
     /, 
    tol: float= .5 , 
    *, 
    interpolate_freq:bool =False, 
    return_freq: bool =False,
    tensor:str ='res', 
    return_data=False,
    to_log10: bool =False, 
    return_qco:bool=False 
    )->Tuple[float, ArrayLike]: 
    """
    Check the quality control in the collection of Z or EDI objects. 
    
    Analyse the data in the EDI collection and return the quality control value.
    It indicates how percentage are the data to be representative.
   
    Parameters 
    ----------
    tol: float, default=.5 
        the tolerance parameter. The value indicates the rate from which the 
        data can be consider as meaningful. Preferably it should be less than
        1 and greater than 0.  Default is ``.5`` means 50 %. Analysis becomes 
        soft with higher `tol` values and severe otherwise. 
        
    interpolate_freq: bool, 
        interpolate the valid frequency after removing the frequency which 
        data threshold is under the ``1-tol``% goodness 
    
    return_freq: bool, default=False 
        returns the interpolated frequency.
        
    return_data: bool, default= False, 
        returns the valid data from up to ``1-tol%`` goodness. 
        
    tensor: str, default='z'  
        Tensor name. Can be [ resistivity|phase|z|frequency]. Impedance is
        used for data quality assessment. 
        
    to_log10: bool, default=True 
       convert the frequency value to log10. 
       
    return qco: bool, default=False, 
       retuns quality control object that wraps all usefull informations after 
       control. The following attributes can be fetched as: 
           
       - rate_: the rate of the quality of the data  
       - component_: The selected component where data is selected for analysis 
         By default used either ``xy`` or ``yx``. 
       - mode_: The :term:`EM` mode. Either the ['TE'|'TM'] modes 
       - freqs_: The valid frequency in the data selected according to the 
         `tol` parameters. Note that if ``interpolate_freq`` is ``True``, it 
         is used instead. 
       - invalid_freqs_: Useless frequency dropped in the data during control 
       - data_: Valid tensor data either in TE or TM mode. 
       
    Returns 
    -------
    Tuple (float  )  or (float, array-like, shape (N, )) or QCo
        - return the quality control value and interpolated frequency if  
         `return_freq`  is set to ``True`` otherwise return the
         only the quality control ratio.
        - return the the quality control object. 
        
    Examples 
    -----------
    >>> import hlearn as wx 
    >>> data = wx.fetch_data ('huayuan', samples =20, return_data =True ,
                              key='raw')
    >>> r,= wx.qc (data)
    r
    Out[61]: 0.75
    >>> r, = wx.qc (data, tol=.2 )
    0.75
    >>> r, = wx.qc (data, tol=.1 )
    
    """
    tol = assert_ratio(tol , bounds =(0, 1), exclude_value ='use lower bound',
                         name ='tolerance', in_percent =True )
    # by default , we used the resistivity tensor and error at TE mode.
    # force using the error when resistivity or phase tensors are supplied 
    tensor = str(tensor).lower() 
    try:
        component, mode ='xy', 'TE'
        ar, f = get2dtensor(z_or_edis_obj_list, tensor =tensor,
                            component =component, return_freqs=True )
    except : 
       component, mode ='yx', 'TM'
       ar, f = get2dtensor(z_or_edis_obj_list, tensor =tensor,
                           return_freqs=True, component =component, 
                           )
       
    # compute the ratio of NaN in axis =0 
    nan_sum  =np.nansum(np.isnan(ar), axis =1) 

    rr= np.around ( nan_sum / ar.shape[1] , 2) 
 
    # compute the ratio ck
    # ck = 1. -    rr[np.nonzero(rr)[0]].sum() / (
    #     1 if len(np.nonzero(rr)[0])== 0 else len(np.nonzero(rr)[0])) 
    # ck =  (1. * len(rr) - len(rr[np.nonzero(rr)[0]]) )  / len(rr)
    
    # using np.nonzero(rr) seems deprecated 
    # ck = 1 - nan_sum[np.nonzero(rr)[0]].sum() / (
    #     ar.shape [0] * ar.shape [1]) 
    ck = 1 - nan_sum[rr[0]].sum() / (
        ar.shape [0] * ar.shape [1]) 
    # now consider dirty data where the value is higher 
    # than the tol parameter and safe otherwise. 
    index = reshape (np.argwhere (rr > tol))
    ar_new = np.delete (rr , index , axis = 0 ) 
    new_f = np.delete (f[:, None], index, axis =0 )
    # interpolate freq 
    if f[0] < f[-1]: 
        f =f[::-1] # reverse the freq array 
        ar_new = ar_new [::-1] # or np.flipud(np.isnan(ar)) 
        
    # get the invalid freqs 
    invalid_freqs= f[ ~np.isin (f, new_f) ]
    
    if interpolate_freq: 
        new_f = np.logspace(
            np.log10(new_f.min()) , 
            np.log10(new_f.max()),
            len(new_f))[::-1]
        # since interpolation is already made in 
        # log10, getback to normal by default 
        # and set off to False
        if not to_log10: 
            new_f = np.power(10, new_f)
            
        to_log10=False  
        
    if to_log10: 
        new_f = np.log10 ( new_f ) 
        
    # for consistency, recovert frequency to array shape 0 
    new_f = reshape (new_f)
    
    # Return frequency if interpolation or frequency conversion
    # is set to True 
    if ( interpolate_freq or to_log10 ): 
        return_freq =True 
    # if return QCobj then block all returns  to True 
    if return_qco: 
        return_freq = return_data = True 
        
    data =[ np.around (ck, 2) ] 
    if return_freq: 
        data += [ new_f ]  
    if return_data :
        data += [ np.delete ( ar, index , axis =0 )] 
        
    data = tuple (data )
    # make QCO object 
    if return_qco: 
        data = Boxspace( **dict (
            tol=tol, 
            tensor = tensor, 
            component_= component, 
            mode_=mode, 
            rate_= float(np.around (ck, 2)), 
            freqs_= new_f , 
            invalid_freqs_=invalid_freqs, 
            data_=  np.delete ( ar, index , axis =0 )
            )
        )
    return data
 
def get_distance(
    x: ArrayLike, 
    y:ArrayLike , *, 
    return_mean_dist:bool =False, 
    is_latlon= False , 
    **kws
    ): 
    """
    Compute distance between points
    
    Parameters
    ------------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    return_mean_dist: bool, default =False, 
       Returns the average value of the distance between different points. 
       
    is_latlon: bool, default=False, 
        Convert `x` and `y` latitude  and longitude coordinates values 
        into UTM before computing the distance. `x`, `y` should be considered 
        as ``easting`` and ``northing`` respectively. 
        
    kws: dict, 
       Keyword arguments passed to :meth:`hlearn.site.Location.to_utm_in`
       
    Returns 
    ---------
    d: Arraylike of shape (N-1) 
      Is the distance between points. 
      
    Examples 
    --------- 
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import get_distance 
    >>> x = np.random.rand (7) *10 
    >>> y = np.abs ( np.random.randn (7) * 12 ) 
    >>> get_distance (x, y) 
    array([ 8.7665511 , 12.47545656,  8.53730212, 13.54998351, 14.0419387 ,
           20.12086781])
    >>> get_distance (x, y, return_mean_dist= True) 
    12.91534996818084
    """
    x, y = _assert_x_y_positions (x, y, is_latlon , **kws  )
    d = np.sqrt( np.diff (x) **2 + np.diff (y)**2 ) 
    
    return d.mean()  if return_mean_dist else d 

def scale_positions (
    x: ArrayLike, 
    y:ArrayLike, 
    *, 
    is_latlon:bool=False, 
    step:float= None, 
    use_average_dist:bool=False, 
    utm_zone:str= None, 
    shift: bool=True, 
    view:bool = False, 
    **kws
    ): 
    """
    Correct the position coordinates. 
     
    By default, it consider `x` and `y` as easting/latitude and 
    northing/longitude coordinates respectively. It latitude and longitude 
    are given, specify the parameter `is_latlon` to ``True``. 
    
    Parameters
    ----------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    is_latlon: bool, default=False, 
       Convert `x` and `y` latitude  and longitude coordinates values 
       into UTM before computing the distance. `x`, `y` should be considered 
       as ``easting`` and ``northing`` respectively. 
           
    step: float, Optional 
       The positions separation. If not given, the average distance between 
       all positions should be used instead. 
    use_average_dist: bool, default=False, 
       Use the distance computed between positions for the correction. 
    utm_zone: str,  Optional (##N or ##S)
       UTM zone in the form of number and North or South hemisphere. For
       instance '10S' or '03N'. Note that if `x` and `y` are UTM coordinates,
       the `utm_zone` must be provide to accurately correct the positions, 
       otherwise the default value ``49R`` should be used which may lead to 
       less accuracy. 
       
    shift: bool, default=True,
       Shift the coordinates from the units of `step`. This is the default 
       behavor. If ``False``, the positions are just scaled. 
    
    view: bool, default=True 
       Visualize the scaled positions 
       
    kws: dict, 
       Keyword arguments passed to :func:`~.get_distance` 
    Returns 
    --------
    xx, yy: Arraylike 1d, 
       The arrays of position correction from `x` and `y` using the 
       bearing. 
       
    See Also 
    ---------
    hlearn.utils.exmath.get_bearing: 
        Compute the  direction of one point relative to another point. 
      
    Examples
    ---------
    >>> from hlearn.utils.exmath import scale_positions 
    >>> east = [336698.731, 336714.574, 336730.305] 
    >>> north = [3143970.128, 3143957.934, 3143945.76]
    >>> east_c , north_c= scale_positions (east, north, step =20, view =True  ) 
    >>> east_c , north_c
    (array([336686.69198337, 336702.53498337, 336718.26598337]),
     array([3143986.09866306, 3143973.90466306, 3143961.73066306]))
    """
    from ..site import Location
    
    msg =("x, y are not in longitude/latitude format  while 'utm_zone' is not"
          " supplied. Correction should be less accurate. Provide the UTM"
          " zone to improve the accuracy.")
    
    if is_latlon: 
        xs , ys = np.array(copy.deepcopy(x)) , np.array(copy.deepcopy(y))

    x, y = _assert_x_y_positions( x, y, islatlon = is_latlon , **kws ) 
    
    if step is None: 
        warnings.warn("Step is not given. Average distance between points"
                      " should be used instead.")
        use_average_dist =True 
    else:  
        d = float (_assert_all_types(step, float, int , objname ='Step (m)'))
    if use_average_dist: 
        d = get_distance(x, y, return_mean_dist=use_average_dist,  **kws) 
        
    # compute bearing. 
    utm_zone = utm_zone or '49R'
    if not is_latlon and utm_zone is None: 
        warnings.warn(msg ) 
    if not is_latlon: 
        xs , ys = Location.to_latlon_in(x, y, utm_zone= utm_zone) 
  
    b = get_bearing((xs[0] , ys[0]) , (xs[-1], ys[-1]),
                    to_deg =False ) # return bearing in rad.
 
    xx = x + ( d * np.cos (b))
    yy = y +  (d * np.sin(b))
    if not shift: 
        xx, *_ = scalePosition(x )
        yy, *_ = scalePosition(y)
        
    if view: 
        state = f"{'scaled' if not shift else 'shifted'}"
        plt.plot (x, y , 'ok-', label =f"Un{state} positions") 
        plt.plot (xx , yy , 'or:', label =f"{state.title()} positions")
        plt.xlabel ('x') ; plt.ylabel ('y')
        plt.legend()
        plt.show () 
        
    return xx, yy 

def _assert_x_y_positions (x, y , islatlon = False, is_utm=True,  **kws): 
    """ Assert the position x and y and return array of x and y  """
    from ..site import Location 
    x = np.array(x, dtype = np.float64) 
    y = np.array(y, np.float64)
    for ii, ar in enumerate ([x, y]):
        if not _is_arraylike_1d(ar):
            raise TypeError (
                f"Expect one-dimensional array for {'x' if ii==0 else 'y'!r}."
                " Got {x.ndim}d.")
        if len(ar) <= 1:
            raise ValueError (f"A singleton array {'x' if ii==0 else 'y'!r} is"
                              " not admitted. Expect at least two points"
                              " A(x1, y1) and B(x2, y2)")
    if islatlon: 
        x , y = Location.to_utm_in(x, y, **kws)
    return x, y 

def get_bearing (latlon1, latlon2,  to_deg = True ): 
    """
    Calculate the bearing between two points. 
     
    A bearing can be defined as  a direction of one point relative 
    to another point, usually given as an angle measured clockwise 
    from north.
    The formula of the bearing :math:`\beta` between two points 1(lat1 , lon1)
    and 2(lat2, lon2) is expressed as below: 
        
    .. math:: 
        \beta = atan2(sin(y_2-y_1)*cos(x_2), cos(x_1)*sin(x_2) – \
                      sin(x_1)*cos(x_2)*cos(y_2-y_1))
     
    where: 
       
       - :math:`x_1`(lat1): the latitude of the first coordinate
       - :math:`y_1`(lon1): the longitude of the first coordinate
       - :math:`x_2`(lat2) : the latitude of the second coordinate
       - :math:`y_2`(lon2): the longitude of the second coordinate
    
    Parameters 
    ----------- 
    latlon: Tuple ( latitude, longitude) 
       A latitude and longitude coordinates of the first point in degree. 
    latlon2: Tuple ( latitude, longitude) 
       A latitude and longitude of coordinates of the second point in degree.  
       
    to_deg: bool, default=True 
       Convert the bearing from radians to degree. 
      
    Returns 
    ---------
    b: Value of bearing in degree ( default). 
    
    See More 
    ----------
    See more details by clicking in the link below: 
        https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
        
    Examples 
    ---------
    >>> from hlearn.utils import get_bearing 
    >>> latlon1 = (28.41196763902007, 109.3328724432221) # (lat, lon) point 1
    >>> latlon2= (28.38756530909265, 109.36931920880758) # (lat, lon) point 2
    >>> get_bearing (latlon1, latlon2 )
    127.26739270447973 # in degree 
    """
    latlon1 = reshape ( np.array ( latlon1, dtype = np.float64)) 
    latlon2 = reshape ( np.array ( latlon2, dtype = np.float64)) 
    
    if len(latlon1) <2 or len(latlon2) <2 : 
        raise ValueError("Wrong coordinates values. Need two coordinates"
                         " (latitude and longitude) of points 1 and 2.")
    lat1 = np.deg2rad (latlon1[0]) ; lon1 = np.deg2rad(latlon1[1])
    lat2 = np.deg2rad (latlon2[0]) ; lon2 = np.deg2rad(latlon2[1])
    
    b = np.arctan2 (
        np.sin(lon2 - lon1 )* np.cos (lat2), 
        np.cos (lat1) * np.sin(lat2) - np.sin (lat1) * np.cos (lat2) * np.cos (lon2 - lon1)
                    )
    if to_deg: 
        # convert bearing to degree and make sure it 
        # is positive between 360 degree 
        b = ( np.rad2deg ( b) + 360 )% 360 
        
    return b 

def find_closest( arr, /, values ): 
    """Get the closest value in array  from given values.
    
    Parameters 
    -----------
    arr : Arraylike  
       Array to find the values 
       
    values: float, arraylike 
    
    Returns
    --------
    closest values in float or array containing in the given array.
    
    Examples
    -----------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import find_closest
    >>> find_closest (  [ 2 , 3, 4, 5] , ( 2.6 , 5.6 )  )
    array([3., 5.])
    >>> find_closest (  np.array ([[2 , 3], [ 4, 5]]), ( 2.6 , 5.6 ) )
    array([3., 5.])
    array([3., 5.])
    """

    arr = is_iterable(arr, exclude_string=True , transform =True  )
    values = is_iterable(values , exclude_string=True  , transform =True ) 
    
    for ar, v in zip ( [ arr, values ], ['array', 'values']): 
        if not _is_numeric_dtype(arr, to_array= True ) :
            raise TypeError(f"Non-numerical {v} are not allowed.")
        
    arr = np.array (arr, dtype = np.float64 )
    values = np.array (values, dtype = np.float64 ) 
    
    # ravel arr if ndim is not one-dimensional 
    arr  = arr.ravel() if arr.ndim !=1 else arr 
    # Could Find the absolute difference with each value   
    # Get the index of the smallest absolute difference. 
    
    # --> Using map is less faster than list comprehension 
    # close = np.array ( list(
    #     map (lambda v: np.abs ( arr - v).argmin(), values )
    #                   ), dtype = np.float64
    #     )
    return np.array ( [
        arr [ np.abs ( arr - v).argmin()] for v in values ]
        )
  
def gradient_descent(
    z: ArrayLike, 
    s:ArrayLike, 
    alpha:float=.01, 
    n_epochs:int= 100,
    kind:str="linear", 
    degree:int=1, 
    raise_warn:bool=False, 
    ): 
    """ Gradient descent algorithm to  fit the best model parameter.
    
    Model can be changed to polynomial if degree is greater than 1. 
    
    Parameters 
    -----------
    z: arraylike, 
       vertical nodes containing the values of depth V
    s: Arraylike, 
       vertical vector containin the resistivity values 
    alpha: float,
       step descent parameter or learning rate. *Default* is ``0.01`
    n_epochs: int, 
       number of iterations. *Default* is ``100``. Can be changed to other values
    kind: str, {"linear", "poly"}, default= 'linear'
      Type of model to fit. Linear model is selected as the default. 
    degree: int, default=1 
       As the linear model is selected as the default since the degree is set 
       to ``1``
    Returns 
    ---------
    - `F`: New model values with the best `W` parameters found.
    - `W`: vector containing the parameters fits 
    - `cost_history`: Containing the error at each Itiretaions. 
        
    Examples 
    -----------
    >>> import numpy as np 
    >>> from hlearn.utils.exmath import gradient_descent
    >>> z= np.array([0, 6, 13, 20, 29 ,39, 49, 59, 69, 89, 109, 129, 
                     149, 179])
    >>> res= np.array( [1.59268,1.59268,2.64917,3.30592,3.76168,
                        4.09031,4.33606, 4.53951,4.71819,4.90838,
          5.01096,5.0536,5.0655,5.06767])
    >>> fz, weights, cost_history = gradient_descent(
        z=z, s=res,n_epochs=10,alpha=1e-8,degree=2)
    >>> import matplotlib.pyplot as plt 
    >>> plt.scatter (z, res)
    >>> plt.plot(z, fz)
    """
    
    #Assert degree
    try :degree= abs(int(degree)) 
    except:raise TypeError(f"Degree is integer. Got {type(degree).__name__!r}")
    
    if degree >1 :
        kind='poly'
        
    kind = str(kind).lower()    
    if kind.lower() =='linear': 
        # block degree to one.
        degree = 1 
    elif kind.find('poly')>=0 : 
        if degree <=1 :
            warnings.warn(
                "Polynomial function expects degree greater than 1."
                f" Got {degree!r}. Value is resetting to minimum equal 2."
                      ) if raise_warn else None 
            degree = 2
    # generate function with degree 
    Z, W = _kind_of_model(degree=degree,  x=z, y=s)
    
    # Compute the gradient descent 
    cost_history = np.zeros(n_epochs)
    s=s.reshape((s.shape[0], 1))
    
    for ii in range(n_epochs): 
        with np.errstate(all='ignore'): # rather than divide='warn'
            #https://numpy.org/devdocs/reference/generated/numpy.errstate.html
            W= W - (Z.T.dot(Z.dot(W)-s)/ Z.shape[0]) * alpha 
            cost_history[ii]= (1/ 2* Z.shape[0]) * np.sum((Z.dot(W) -s)**2)
       
    # Model function F= Z.W where `Z` id composed of vertical nodes 
    # values and `bias` columns and `W` is weights numbers.
    F= Z.dot(W) # model(Z=Z, W=W)     # generate the new model with the best weights 
             
    return F,W, cost_history

def _kind_of_model(degree, x, y) :
    """ 
    An isolated part of gradient descent computing. 
    Generate kind of model. If degree is``1`` The linear subset 
    function will use. If `degree` is greater than 2,  Matrix will 
    generate using the polynomail function.
     
    :param x: X values must be the vertical nodes values 
    :param y: S values must be the resistivity of subblocks at node x 
    
    """
    c= []
    deg = degree 
    w = np.zeros((degree+1, 1)) # initialize weights 
    
    def init_weights (x, y): 
        """ Init weights by calculating the scope of the function along 
         the vertical nodes axis for each columns. """
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', 
                                    category=RuntimeWarning)
            for j in range(x.shape[1]-1): 
                a= (y.max()-y.min())/(x[:, j].max()-x[:, j].min())
                w[j]=a
            w[-1] = y.mean()
        return w   # return weights 

    for i in range(degree):
        c.append(x ** deg)
        deg= deg -1 

    if len(c)> 1: 
        x= concat_array_from_list(c, concat_axis=1)
        x= np.concatenate((x, np.ones((x.shape[0], 1))), axis =1)

    else: x= np.vstack((x, np.ones(x.shape))).T # initialize z to V*2

    w= init_weights(x=x, y=y)
    return x, w  # Return the matrix x and the weights vector w 
    
























   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    