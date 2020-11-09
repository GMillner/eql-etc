# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:06:51 2020

@author: g.millner
"""

__author__ = "Gerfried Millner"
__version__ = "0.2.0"
__date__ = "28.07.2020"
__email__ = "g.millner@gmx.at" 
__status__ = "Development"

import numpy as np
import sympy
import pickle

from sympy import symbols, Function

import pandas as pd
from os import path, walk

# import tensorflow as tf

import math

from collections import namedtuple

import sys
import os
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(file_path, '..')
if lib_path not in sys.path:
    sys.path.append(lib_path)
    
# from src.evaluation import get_symbol_list, symbolic_matmul_and_bias, symbolic_eql_layer, round_sympy_expr, proper_simplify
# from src.timeout import time_limit, TimeoutException

# from src.utils import update_runtime_params, get_div_thresh_fn
# from src.data_utils import extract_metadata

# import src.EQL_Layer_tf as eql

# from src.config import *


#print(lib_path)


def _np_from_npy_formulas(directory, filename='formula_0.npy'):
    """ Returns a generator producing list for each npy file with given filename in given directory (recursively).
    
    Args:
        directory: path to directory, where npy files are stored (in subfolders), string
        filename: name of npy file, string
        
    Returns:
        formula_list: list of all npy files found in the subdirectories, list    
    """
    formula_list = []
    formula_dir = []
    for root, dirs, files in walk(directory):
        if filename in files:
            #print('here: ', root)
            formula_dir.append(root)
            formula_list.append(np.load(path.join(root, filename), allow_pickle=True))
            
    return formula_list, formula_dir



if __name__ == '__main__':
    
    # example | print formulas from model selection results 
    
    path_model_sel = r''
    
    formula_list, formula_dir = _np_from_npy_formulas(path_model_sel, 'formula_0.npy')
    
    for i in range(len(formula_list)):
        print(i)
        print(formula_dir[i])
        print(formula_list[i])
    
    

    
    
    


