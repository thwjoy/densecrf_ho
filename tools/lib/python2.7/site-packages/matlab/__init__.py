# Copyright 2014 MathWorks, Inc.
"""
Array interface between Python and the MATLAB runtime.

This package defines classes and exceptions that create and manage
multidimensional arrays in Python that are passed between Python and the
MATLAB runtime. The arrays, while similar to Python sequences,
have different behaviours.

Modules
-------
    * mlarray - type-specific multidimensional array classes for working
    with the MATLAB runtime
    * mlexceptions - exceptions raised manipulating mlarray objects
"""

import os
import sys

_package_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(_package_folder)

from mlarray import double, single, uint8, int8, uint16, \
    int16, uint32, int32, uint64, int64, logical
from mlexceptions import ShapeError as ShapeError
from mlexceptions import SizeError as SizeError
