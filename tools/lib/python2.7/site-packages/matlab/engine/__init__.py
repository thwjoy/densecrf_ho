#Copyright 2014 MathWorks, Inc.

"""
The MATLAB Engine enables you to call any MATLAB statement either synchronously
or asynchronously.  With synchronous execution, the invocation of a MATLAB
statement returns the result after the call finishes.  With asynchronous
execution, the invocation of a MATLAB statement returns a FutureResult object
immediately.  You can call its "done" function to check if the call has finished,
and its "result" function to obtain the actual result of the MATLAB statement.

This example shows how to call a MATLAB function:

>>> import matlab.engine
>>> eng = matlab.engine.start_matlab()
>>> eng.sqrt(4.0)
2.0
>>> eng.exit()
"""


import os
import sys
import importlib
import atexit
import weakref

_supported_versions = ['2_7', '3_3', '3_4']
_ver = sys.version_info
_version = '{0}_{1}'.format(_ver[0], _ver[1])
_PYTHONVERSION = None

if _version in _supported_versions:
    _PYTHONVERSION = _version
else:
    raise EnvironmentError("Python %s is not supported." % _version)

_module_folder = os.path.dirname(os.path.realpath(__file__))
_arch_filename = _module_folder+os.sep+"_arch.txt"
 
try:
    pythonengine = importlib.import_module("matlabengineforpython"+_PYTHONVERSION)
except:
    try:
        _arch_file = open(_arch_filename,'r')
        _lines = _arch_file.readlines()
        [_arch, _bin_dir,_engine_dir] = [x.rstrip() for x in _lines if x.rstrip() != ""]
        _arch_file.close()
        sys.path.insert(0,_engine_dir)

        _envs = {'win32': 'PATH', 'win64': 'PATH'}
        if _arch in _envs:
            if _envs[_arch] in os.environ:
                _env = os.environ[_envs[_arch]]
                os.environ[_envs[_arch]] = _bin_dir+os.pathsep+os.environ[_envs[_arch]]
            else:
                os.environ[_envs[_arch]] = _bin_dir
        pythonengine = importlib.import_module("matlabengineforpython"+_PYTHONVERSION)
    except:
        raise EnvironmentError('The installation of MATLAB Engine for Python is '
                                'corrupted.  Please reinstall it or contact '
                                'MathWorks Technical Support for assistance.')

from matlab.engine.engineerror import RejectedExecutionError
from matlab.engine.futureresult import FutureResult
from matlab.engine.enginesession import EngineSession
from matlab.engine.matlabengine import MatlabEngine

_session = EngineSession()
_engines = []
    
def start_matlab(option="-nodesktop"):
    """
    Start the MATLAB Engine.  This function creates an instance of the
    MatlabEngine class.  The local version of MATLAB will be launched
    with the "-nodesktop" argument.

    Please note the invocation of this function is synchronous, which
    means it only returns after MATLAB launches.
    
    Parameters
        option - MATLAB startup option.
                
    Returns
        MatlabEngine - this object can be used to evaluate MATLAB
        statements.

    Raises
        EngineError - if MATLAB can't be started.
    """
    if not isinstance(option, str):
        raise TypeError('MATLAB startup option should be str')
    eng = MatlabEngine(option)
    _engines.append(weakref.ref(eng))
    return eng

@atexit.register
def __exit_engines():
    for eng in _engines:
        if eng() is not None:
            eng().exit()
    _session.release()