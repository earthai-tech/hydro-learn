# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import ( 
    annotations , 
    print_function 
    )
import os
import hlearn
import shutil 
import pathlib
import warnings
from six.moves import urllib 

import numpy as np 
import pandas as pd 

from .._typing import ( 
    Any, 
    List, 
    NDArray, 
    DataFrame, 
    Tuple,
    Optional,
    ArrayLike, 
    DType, 
    )
from .funcutils import ( 
    is_iterable, 
    ellipsis2false,
    smart_format,
    sPath, 
    to_numeric_dtypes, 
    _assert_all_types, 

    )
from .gistools import ( 
    HAS_GDAL, 
    convert_position_float2str, 
    project_point_utm2ll, 
    convert_position_str2float, 
    assert_lat_value, 
    assert_lon_value, 
    utm_to_ll, 
    
    )
from ..property import Config
from ._dependency import ( 
    import_optional_dependency 
    )
from .validator import array_to_frame 
from ..exceptions import FileHandlingError

def read_data (
    f: str|pathlib.PurePath, 
    sanitize: bool= ..., 
    reset_index: bool=..., 
    comments: str="#", 
    delimiter: str=None, 
    columns: List[str]=None,
    npz_objkey: str= None, 
    verbose: bool= ..., 
    **read_kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a data frame.  
    
    Parameters 
    -----------
    f: str, Path-like object 
       File path or Pathlib object. Must contain a valid file name  and 
       should be a readable file or url 
        
    sanitize: bool, default=False, 
       Push a minimum sanitization of the data such as: 
           - replace a non-alphabetic column items with a pattern '_' 
           - cast data values to numeric if applicable 
           - drop full NaN columns and rows in the data 
           
    reset_index: bool, default=False, 
      Reset index if full NaN columns are dropped after sanitization. 
      
      .. versionadded:: 0.2.5
          Apply minimum data sanitization after reading data. 
     
    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. 

    delimiter: str, optional
       The character used to separate the values. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. The default is whitespace.

    npz_objkey: str, optional 
       Dataset key to indentify array in multiples array storages in '.npz' 
       format.  If key is not set during 'npz' storage, ``arr_0`` should 
       be used. 
      
       .. versionadded:: 0.2.7 
          Capable to read text and numpy formats ('.npy' and '.npz') data. Note
          that when data is stored in compressed ".npz" format, provided the 
          '.npz' object key  as argument of parameter `npz_objkey`. If None, 
          only the first array should be read and ``npz_objkey='arr_0'``. 
          
    verbose: bool, default=0 
       Outputs message for user guide. 
       
    read_kws: dict, 
       Additional keywords arguments passed to pandas readable file keywords. 
        
    Returns 
    -------
    f: :class:`pandas.DataFrame` 
        A dataframe with head contents by default.  
        
    See Also 
    ---------
    np.loadtxt: 
        load text file.  
    np.load 
       Load uncompressed or compressed numpy `.npy` and `.npz` formats. 
    hlearn.utils.baseutils.save_or_load: 
        Save or load numpy arrays.
       
    """
    def min_sanitizer ( d, /):
        """ Apply a minimum sanitization to the data `d`."""
        return to_numeric_dtypes(
            d, sanitize_columns= True, 
            drop_nan_columns= True, 
            reset_index=reset_index, 
            verbose = verbose , 
            fill_pattern='_', 
            drop_index = True
            )
    sanitize, reset_index, verbose = ellipsis2false (
        sanitize, reset_index, verbose )
    
    if ( isinstance ( f, str ) 
            and str(os.path.splitext(f)[1]).lower()in (
                '.txt', '.npy', '.npz')
            ): 
        f = save_or_load(f, task = 'load', comments=comments, 
                         delimiter=delimiter )
        # if extension is .npz
        if isinstance(f, np.lib.npyio.NpzFile):
            npz_objkey = npz_objkey or "arr_0"
            f = f[npz_objkey] 

        if columns is not None: 
            columns = is_iterable(columns, exclude_string= True, 
                                  transform =True, parse_string =True 
                                  )
            if len( columns )!= f.shape [1]: 
                warnings.warn(f"Columns expect {f.shape[1]} attributes."
                              f" Got {len(columns)}")
            
        f = pd.DataFrame(f, columns=columns )
        
    if isinstance (f, pd.DataFrame): 
        if sanitize: 
            f = min_sanitizer (f)
        return  f 
    
    cpObj= Config().parsers 
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(f"Can only parse the {smart_format(cpObj.keys(), 'or')} files"
                        )
    try : 
        f = cpObj[ex](f, **read_kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except BaseException as e : 
        raise FileHandlingError (
            f"Cannot parse the file : {os.path.basename (f)!r}. "+  str(e))
    if sanitize: 
        f = min_sanitizer (f)
        
    return f 
    
def _check_readable_file (f): 
    """ Return file name from path objects """
    msg =(f"Expects a Path-like object or URL. Please, check your"
          f" file: {os.path.basename(f)!r}")
    if not os.path.isfile (f): # force pandas read html etc 
        if not ('http://'  in f or 'https://' in f ):  
            raise TypeError (msg)
    elif not isinstance (f,  (str , pathlib.PurePath)): 
         raise TypeError (msg)
    if isinstance(f, str): f =f.strip() # for consistency 
    return f 
def _is_readable (
        f:str, 
        *, 
        as_frame:bool=False, 
        columns:List[str]=None,
        input_name='f', 
        **kws
 ) -> DataFrame: 
    """ Assert and read specific files and url allowed by the package
    
    Readable files are systematically convert to a pandas frame.  
    
    Parameters 
    -----------
    f: Path-like object -Should be a readable files or url  
    columns: str or list of str 
        Series name or columns names for pandas.Series and DataFrame. 
        
    to_frame: str, default=False
        If ``True`` , reconvert the array to frame using the columns orthewise 
        no-action is performed and return the same array.
    input_name : str, default=""
        The data name used to construct the error message. 
        
    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.
        If ``ignore``, warnings silence mode is triggered.
    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.
        
    force:bool, default=False
        Force conversion array to a frame is columns is not supplied.
        Use the combinaison, `input_name` and `X.shape[1]` range.
        
    kws: dict, 
        Pandas readableformats additional keywords arguments. 
    Returns
    ---------
    f: pandas dataframe 
         A dataframe with head contents... 
    
    """
    if hasattr (f, '__array__' ) : 
        f = array_to_frame(
            f, 
            to_frame= True , 
            columns =columns, 
            input_name=input_name , 
            raise_exception= True, 
            force= True, 
            )
        return f 

    cpObj= Config().parsers 
    
    f= _check_readable_file(f)
    _, ex = os.path.splitext(f) 
    if ex.lower() not in tuple (cpObj.keys()):
        raise TypeError(
            f"Can only parse the {smart_format(cpObj.keys(), 'or')}"
            f" files not {ex!r}.")
    try : 
        f = cpObj[ex](f, **kws)
    except FileNotFoundError:
        raise FileNotFoundError (
            f"No such file in directory: {os.path.basename (f)!r}")
    except: 
        raise FileHandlingError (
            f" Can not parse the file : {os.path.basename (f)!r}")

    return f 


def array2hdf5 (
    filename: str, /, 
    arr: NDArray=None , 
    dataname: str='data',  
    task: str='store', 
    as_frame: bool =..., 
    columns: List[str, ...]=None, 
)-> NDArray | DataFrame: 
    """ Load or write array to hdf5
    
    Parameters 
    -----------
    arr: Arraylike ( m_samples, n_features) 
      Data to load or write 
    filename: str, 
      Hdf5 disk file name whether to write or to load 
    task: str, {"store", "load", default='store'}
       Action to perform. user can use ['write'|'store'] interchnageably. Both 
       does the same task. 
    as_frame: bool, default=False 
       Concert loaded array to data frame. `Columns` can be supplied 
       to construct the datafame. 
    columns: List, Optional 
       Columns used to construct the dataframe. When its given, it must be 
       consistent with the shape of the `arr` along axis 1 
       
    Returns 
    ---------
    None| data: ArrayLike or pd.DataFrame 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from hlearn.utils.baseutils import array2hdf5
    >>> data = np.random.randn (100, 27 ) 
    >>> array2hdf5 ('test.h5', data   )
    >>> load_data = array2hdf5 ( 'test.h5', data, task ='load')
    >>> load_data.shape 
    Out[177]: (100, 27)
    """
    import_optional_dependency("h5py")
    import h5py 
    
    arr = is_iterable( arr, exclude_string =True, transform =True )
    act = hlearn.deephlearn(task)
    task = str(task).lower().strip() 
    
    if task in ("write", "store"): 
        task ='store'
    assert task in {"store", "load"}, ("Expects ['store'|'load'] as task."
                                         f" Got {act!r}")
    # for consistency 
    arr = np.array ( arr )
    h5fname = str(filename).replace ('.h5', '')
    if task =='store': 
        if arr is None: 
            raise TypeError ("Array cannot be None when the task"
                             " consists to write a file.")
        with h5py.File(h5fname + '.h5', 'w') as hf:
            hf.create_dataset(dataname,  data=arr)
            
    elif task=='load': 
        with h5py.File(h5fname +".h5", 'r') as hf:
            data = hf[dataname][:]
            
        if  ellipsis2false( as_frame )[0]: 
            data = pd.DataFrame ( data , columns = columns )
            
    return data if task=='load' else None 
   
def lowertify (*values, strip = True, return_origin: bool =... ): 
    """ Strip and convert value to lowercase. 
    
    :param value: str , value to convert 
    :return: value in lowercase and original value. 
    
    :Example: 
        >>> from hlearn.utils.baseutils import lowertify 
        >>> lowertify ( 'KIND')
        Out[19]: ('kind',)
        >>> lowertify ( "KIND", return_origin =True )
        Out[20]: (('kind', 'KIND'),)
        >>> lowertify ( "args1", 120 , 'ArG3') 
        Out[21]: ('args1', '120', 'arg3')
        >>> lowertify ( "args1", 120 , 'ArG3', return_origin =True ) 
        Out[22]: (('args1', 'args1'), ('120', 120), ('arg3', 'ArG3'))
        >>> (kind, kind0) , ( task, task0 ) = lowertify(
            "KIND", "task ", return_origin =True )
        >>> kind, kind0, task, task0 
        Out[23]: ('kind', 'KIND', 'task', 'task ')
        """
    raw_values = hlearn.deephlearn(values ) 
    values = [ str(val).lower().strip() if strip else str(val).lower() 
              for val in values]

    return tuple (zip ( values, raw_values)) if ellipsis2false (
        return_origin)[0]  else tuple (values)

def save_or_load(
    fname:str, /,
    arr: NDArray=None,  
    task: str='save', 
    format: str='.txt', 
    compressed: bool=...,  
    comments: str="#",
    delimiter: str=None, 
    **kws 
): 
    """Save or load Numpy array. 
    
    Parameters 
    -----------
    fname: file, str, or pathlib.Path
       File or filename to which the data is saved. 
       - >.npy , .npz: If file is a file-object, then the filename is unchanged. 
       If file is a string or Path, a .npy extension will be appended to the 
       filename if it does not already have one. 
       - >.txt: If the filename ends in .gz, the file is automatically saved in 
       compressed gzip format. loadtxt understands gzipped files transparently.
       
    arr: 1D or 2D array_like
      Data to be saved to a text, npy or npz file.
      
    task: str {"load", "save"}
      Action to perform. "Save" for storing file into the format 
      ".txt", "npy", ".npz". "load" for loading the data from storing files. 
      
    format: str {".txt", ".npy", ".npz"}
       The kind of format to save and load.  Note that when loading the 
       compressed data saved into `npz` format, it does not return 
       systematically the array rather than `np.lib.npyio.NpzFile` files. 
       Use either `files` attributes to get the list of registered files 
       or `f` attribute dot the data name to get the loaded data set. 

    compressed: bool, default=False 
       Compressed the file especially when file format is set to `.npz`. 

    comments: str or sequence of str or None, default='#'
       The characters or list of characters used to indicate the start 
       of a comment. None implies no comments. For backwards compatibility, 
       byte strings will be decoded as 'latin1'. This is useful when `fname`
       is in `txt` format. 
      
     delimiter: str,  optional
        The character used to separate the values. For backwards compatibility, 
        byte strings will be decoded as 'latin1'. The default is whitespace.
        
    **kws: np.save ,np.savetext,  np.load , np.loadtxt 
       Additional keywords arguments for saving and loading data. 
       
    Return 
    ------
    None| data: ArrayLike 
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from hlearn.utils.baseutils import save_or_load 
    >>> data = np.random.randn (2, 7)
    >>> # save to txt 
    >>> save_or_load ( "test.txt" , data)
    >>> save_or_load ( "test",  data, format='.npy')
    >>> save_or_load ( "test",  data, format='.npz')
    >>> save_or_load ( "test_compressed",  data, format='.npz', compressed=True )
    >>> # load files 
    >>> save_or_load ( "test.txt", task ='load')
    Out[36]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test.npy", task ='load')
    Out[39]: array([-2.34162127,  0.55093818])
    >>> save_or_load ( "test.npz", task ='load')
    <numpy.lib.npyio.NpzFile at 0x1b0821870a0>
    >>> npzo = save_or_load ( "test.npz", task ='load')
    >>> npzo.files
    Out[44]: ['arr_0']
    >>> npzo.f.arr_0
    Out[45]: 
    array([[ 0.69265852,  0.67829574,  2.09023489, -2.34162127,  0.48689125,
            -0.04790965,  1.36510779],
           [-1.38349568,  0.63050939,  0.81771051,  0.55093818, -0.43066737,
            -0.59276321, -0.80709192]])
    >>> save_or_load ( "test_compressed.npz", task ='load')
    ...
    """
    r_formats = {"npy", "txt", "npz"}
   
    (kind, kind0), ( task, task0 ) = lowertify(
        format, task, return_origin =True )
    
    assert  kind.replace ('.', '') in r_formats, (
        f"File format expects {smart_format(r_formats, 'or')}. Got {kind0!r}")
    kind = '.' + kind.replace ('.', '')
    assert task in {'save', 'load'}, ( 
        "Wrong task {task0!r}. Valid tasks are 'save' or 'load'") 
    
    save= {'.txt': np.savetxt, '.npy':np.save,  
           ".npz": np.savez_compressed if ellipsis2false(
               compressed)[0] else np.savez 
           }
    if task =='save': 
        arr = np.array (is_iterable( arr, exclude_string= True, 
                                    transform =True ))
        save.get(kind) (fname, arr, **kws )
        
    elif task =='load': 
         ext = os.path.splitext(fname)[1].lower() 
         if ext not in (".txt", '.npy', '.npz', '.gz'): 
             raise ValueError ("Unrecognized file format {ext!r}."
                               " Expect '.txt', '.npy', '.gz' or '.npz'")
         if ext in ('.txt', '.gz'): 
            arr = np.loadtxt ( fname , comments= comments, 
                              delimiter= delimiter,   **kws ) 
         else : 
            arr = np.load(fname,**kws )
         
    return arr if task=='load' else None 
 
#XXX TODO      
def request_data (
    url:str, /, 
    task: str='get',
    data: Any=None, 
    as_json: bool=..., 
    as_text: bool = ..., 
    stream: bool=..., 
    raise_status: bool=..., 
    save2file: bool=..., 
    filename:str =None, 
    **kws
): 
    """ Fetch remotely data
 
    Request data remotely 
    https://docs.python-requests.org/en/latest/user/quickstart/#raw-response-content
    
    
    r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    r.status_code
    200
    r.headers['content-type']
    'application/json; charset=utf8'
    r.encoding
    'utf-8'
    r.text
    '{"type":"User"...'
    r.json()
    {'private_gists': 419, 'total_private_repos': 77, ...}
    
    """
    import_optional_dependency('requests' ) 
    import requests 
    
    as_text, as_json, stream, raise_status, save2file = ellipsis2false(
        as_text, as_json,  stream, raise_status , save2file)
    
    if task=='post': 
        r = requests.post(url, data =data , **kws)
    else: r = requests.get(url, stream = stream , **kws)
    
    if save2file and stream: 
        with open(filename, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
    if raise_status: 
        r.raise_for_status() 
        
    return r.text if as_text else ( r.json () if as_json else r )

def get_remote_data(
    rfile:str, /,  
    savepath: str=None, 
    raise_exception: bool =True
): 
    """ Try to retrieve data from remote.
    
    Parameters 
    -------------
    rfile: str or PathLike-object 
       Full path to the remote file. It can be the path to the repository 
       root toward the file name. For instance, to retrieve the file 
       ``'AGSO.csv'`` which is located in ``hlearn/etc/`` directory then the 
       full path should be ``'hlearn/etc/AGSO.csv'``
        
    savepath: str, optional 
       Full path to place where to downloaded files should be located. 
       If ``None`` data is saved to the current directory.
     
    raise_exception: bool, default=True 
      raise exception if connection failed. 
      
    Returns 
    ----------
    status: bool, 
      ``False`` for failure and ``True`` otherwise i.e. successfully 
       downloaded. 
       
    """
    connect_reason ="""\
    ConnectionRefusedError: No connection could  be made because the target 
    machine actively refused it.There are some possible reasons for that:
     1. Server is not running as well. Hence it won't listen to that port. 
         If it's a service you may want to restart the service.
     2. Server is running but that port is blocked by Windows Firewall
         or other firewall. You can enable the program to go through 
         firewall in the inbound list.
    3. there is a security program on your PC, i.e a Internet Security 
        or Antivirus that blocks several ports on your PC.
    """  
    #git_repo , git_root= AGSO_PROPERTIES['GIT_REPO'], AGSO_PROPERTIES['GIT_ROOT']
    # usebar bar progression
    print(f"---> Please wait while fetching {rfile!r}...")
    try: import_optional_dependency ("tqdm")
    except:pbar = range(3) 
    else: 
        import tqdm  
        data =os.path.splitext( os.path.basename(rfile))[0]
        pbar = tqdm.tqdm (range(3), ascii=True, desc =f'get-{data}', ncols =107)
    status=False
    root, rfile  = os.path.dirname(rfile), os.path.basename(rfile)
    for k in pbar:
        try :
            urllib.request.urlretrieve(root,  rfile )
        except: 
            try :
                with urllib.request.urlopen(root) as response:
                    with open( rfile,'wb') as out_file:
                        data = response.read() # a `bytes` object
                        out_file.write(data)
            except TimeoutError: 
                if k ==2: 
                    print("---> Established connection failed because"
                       "connected host has failed to respond.")
            except:pass 
        else : 
            status=True
            break
        try: pbar.update (k)
        except: pass 
    
    if status: 
        try: pbar.update (3)
        except:pass
        print(f"\n---> Downloading {rfile!r} was successfully done.")
    else: 
        print(f"\n---> Failed to download {rfile!r}.")
    # now move the file to the right place and create path if dir not exists
    if savepath is not None: 
        if not os.path.isdir(savepath): 
            sPath (savepath)
        shutil.move(os.path.realpath(rfile), savepath )
        
    if not status:
        if raise_exception: 
            raise ConnectionRefusedError(connect_reason.replace (
                "ConnectionRefusedError:", "") )
        else: print(connect_reason )
    
    return status
      
def makeCoords(
  reflong: str | Tuple[float], 
  reflat: str | Tuple[float], 
  nsites: int ,  
  *,  
  r: int =45.,
  utm_zone: Optional[str] =None,   
  step: Optional[str|float] ='1km', 
  order: str = '+', 
  todms: bool =False, 
  is_utm: bool  =False,
  raise_warning: bool=True, 
  **kws
)-> Tuple[ArrayLike[DType[float]]]: 
    """ Generate multiple stations coordinates (longitudes, latitudes)
    from a reference station/site.
    
    One degree of latitude equals approximately 364,000 feet (69 miles), 
    one minute equals 6,068 feet (1.15 miles), and one-second equals 101 feet.
    One-degree of longitude equals 288,200 feet (54.6 miles), one minute equals
    4,800 feet (0.91 mile) , and one second equals 80 feet. Illustration showing
    longitude convergence. (1 feet ~=0.3048 meter)
    
    Parameters 
    ----------
    reflong: float or string or list of [start, stop]
        Reference longitude  in degree decimal or in DD:MM:SS for the first 
        site considered as the origin of the landmark.
        
    reflat: float or string or list of [start, stop]
        Reference latitude in degree decimal or in DD:MM:SS for the reference  
        site considered as the landmark origin. If value is given in a list, 
        it can contain the start point and the stop point. 
        
    nsites: int or float 
        Number of site to generate the coordinates onto. 
        
    r: float or int 
        The rotate angle in degrees. Rotate the angle features the direction
        of the projection line. Default value is ``45`` degrees. 
        
    step: float or str 
        Offset or the distance of seperation between different sites in meters. 
        If the value is given as string type, except the ``km``, it should be 
        considered as a ``m`` value. Only meters and kilometers are accepables.
        
    order: str 
        Direction of the projection line. By default the projected line is 
        in ascending order i.e. from SW to NE with angle `r` set to ``45``
        degrees. Could be ``-`` for descending order. Any other value should 
        be in ascending order. 
    
    is_utm: bool, 
        Consider the first two positional arguments as UTM coordinate values. 
        This is an alternative way to assume `reflong` and `reflat` are UTM 
        coordinates 'easting'and 'northing` by default. If `utm2deg` is ``False``, 
        any value greater than 180 degrees for longitude and 90 degrees for 
        latitude will raise an error. Default is ``False``.
        
    utm_zone: string (##N or ##S)
        utm zone in the form of number and North or South hemisphere, 10S or 03N
        Must be given if `utm2deg` is set to ``True``. 
                      
    todms: bool 
        Convert the degree decimal values into the DD:MM:SS. Default is ``False``. 
        
    raise_warning: bool, default=True, 
        Raises warnings if GDAL is not set or the coordinates accurately status.
    
    kws: dict, 
        Additional keywords of :func:`.gistools.project_point_utm2ll`. 
        
    Returns 
    -------
        Tuple of  generated projected coordinates longitudes and latitudes
        either in degree decimals or DD:MM:SS
        
    Notes 
    ------
    The distances vary. A degree, minute, or second of latitude remains 
    fairly constant from the equator to the poles; however a degree, minute,
    or second of longitude can vary greatly as one approaches the poles
    and the meridians converge.
        
    References 
    ----------
    https://math.answers.com/Q/How_do_you_convert_degrees_to_meters
    
    Examples 
    --------
    >>> from watex.utils.coreutils import makeCoords 
    >>> rlons, rlats = makeCoords('110:29:09.00', '26:03:05.00', 
    ...                                     nsites = 7, todms=True)
    >>> rlons
    ... array(['110:29:09.00', '110:29:35.77', '110:30:02.54', '110:30:29.30',
           '110:30:56.07', '110:31:22.84', '110:31:49.61'], dtype='<U12')
    >>> rlats 
    ... array(['26:03:05.00', '26:03:38.81', '26:04:12.62', '26:04:46.43',
           '26:05:20.23', '26:05:54.04', '26:06:27.85'], dtype='<U11')
    >>> rlons, rlats = makeCoords ((116.7, 119.90) , (44.2 , 40.95),
                                            nsites = 238, step =20. ,
                                            order = '-', r= 125)
    >>> rlons 
    ... array(['119:54:00.00', '119:53:11.39', '119:52:22.78', '119:51:34.18',
           '119:50:45.57', '119:49:56.96', '119:49:08.35', '119:48:19.75',
           ...
           '116:46:03.04', '116:45:14.43', '116:44:25.82', '116:43:37.22',
           '116:42:48.61', '116:42:00.00'], dtype='<U12')
    >>> rlats 
    ... array(['40:57:00.00', '40:57:49.37', '40:58:38.73', '40:59:28.10',
           '41:00:17.47', '41:01:06.84', '41:01:56.20', '41:02:45.57',
           ...
       '44:07:53.16', '44:08:42.53', '44:09:31.90', '44:10:21.27',
       '44:11:10.63', '44:12:00.00'], dtype='<U11')
    
    """  
    def assert_ll(coord):
        """ Assert coordinate when the type of the value is string."""
        try: coord= float(coord)
        except ValueError: 
            if ':' not in coord: 
                raise ValueError(f'Could not convert value to float: {coord!r}')
            else : 
                coord = convert_position_str2float(coord)
        return coord
    
    xinf, yinf = None, None 
    
    nsites = int(_assert_all_types(nsites,int, float)) 
    if isinstance (reflong, (list, tuple, np.ndarray)): 
        reflong , xinf, *_ = reflong 
    if isinstance (reflat, (list, tuple, np.ndarray)): 
        reflat , yinf, *_ = reflat 
    step=str(step).lower() 
    if step.find('km')>=0: # convert to meter 
        step = float(step.replace('km', '')) *1e3 
    elif step.find('m')>=0: step = float(step.replace('m', '')) 
    step = float(step) # for consistency 
    
    if str(order).lower() in ('descending', 'down', '-'): order = '-'
    else: order ='+'
    # compute length of line using the reflong and reflat
    # the origin of the landmark is x0, y0= reflong, reflat
    x0= assert_ll(reflong) if is_utm else assert_ll(
        assert_lon_value(reflong))
    y0= assert_ll(reflat) if is_utm else assert_ll(
        assert_lat_value(reflat))
    
    xinf = xinf or x0  + (np.sin(np.deg2rad(r)) * step * nsites
                          ) / (364e3 *.3048) 
    yinf = yinf or y0 + (np.cos(np.deg2rad(r)) * step * nsites
                         ) /(2882e2 *.3048)
    
    reflon_ar = np.linspace(x0 , xinf, nsites ) 
    reflat_ar = np.linspace(y0, yinf, nsites)
    #--------------------------------------------------------------------------
    # r0 = np.sqrt(((x0-xinf)*364e3 *.3048)**2 + ((y0 -yinf)*2882e2 *.3048)**2)
    # print('recover distance = ', r0/nsites )
    #--------------------------------------------------------------------------
    if is_utm : 
        if utm_zone is None: 
            raise TypeError("Please provide your UTM zone e.g.'10S' or '03N' !")
        lon = np.zeros_like(reflon_ar) 
        lat = lon.copy() 
        
        for kk , (lo, la) in enumerate (zip( reflon_ar, reflat_ar)): 
            try : 
                with warnings.catch_warnings(): # ignore multiple warnings 
                    warnings.simplefilter('ignore')
                    lat[kk], lon[kk] = project_point_utm2ll(
                        easting= la, northing=lo, utm_zone=utm_zone, **kws)
            except : 
                lat[kk], lon[kk] = utm_to_ll(
                    23, northing=lo, easting=la, zone=utm_zone)
                
        if not HAS_GDAL : 
            if raise_warning:
                warnings.warn("It seems GDAL is not set! will use the equations"
                              " from USGS Bulletin 1532. Be aware, the positionning" 
                              " is less accurate than using GDAL.")
        
        if raise_warning:
            warnings.warn("By default,'easting/northing' are assumed to"
                          " fit the 'longitude/latitude' respectively.") 
        
        reflat_ar, reflon_ar = lat , lon 
    
    if todms:
       reflat_ar = np.array(list(
           map(lambda l: convert_position_float2str(float(l)), reflat_ar)))
       reflon_ar = np.array(list(
           map(lambda l: convert_position_float2str(float(l)), reflon_ar)))
       
    return (reflon_ar , reflat_ar ) if order =='+' else (
        reflon_ar[::-1] , reflat_ar[::-1] )      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    