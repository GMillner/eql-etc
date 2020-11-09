"""
script to create pandas Dataframes with all results from multiple runs of the EQL stored inside.
"""

__author__ = "Gerfried Millner (GMi)" 
__version__ = "1.2.0" 
__date__ = "07.09.2020"
__email__ = "g.millner@gmx.at" 
__status__ = "Development" 

import numpy as np
#from matplotlib import pyplot as plt
import pandas
import sympy

import pandas as pd
from os import path, walk

# import holoviews as hv
# import bokeh
# from bokeh.io import show

# from holoviews import opts
# hv.extension('bokeh','matplotlib')

from graphviz import Source

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 15),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


################### functions from original EQL - modified ########################

def select_instance(df=None, file=None, use_extrapolation=True):
    """
    Expects a file with one row per network and columns reporting the parameters and complexity and performance
    First line should be the column names, col1 col2 col3..., then one additional comments line which can be empty.
    Third line should be the values for each column.
    :param df: pandas dataframe containing data about model performance
    :param file: file containing data about model performance, only used if dataframe is none
    :param use_extrapolation: flag to determine if extrapolation data should be used
    :return: pandas dataframe containing id and performance data of best model.
    """
    if df is not None and file is not None:
        raise ValueError('Both results_df and file specified. Only specify one.')
    if df is None:
        if file is None:
            raise ValueError('Either results_df or file have to be specified.')
        df = pd.read_csv(file)
    if 'extr_error' in df.keys():
        extr_available = not df['extr_error'].isnull().values.any()
    else:
        extr_available = False
    if use_extrapolation and not extr_available:
        raise ValueError("use_extrapolation flag is set to True but no extrapolation results were found.")

    if use_extrapolation:
        df['extr_normed'] = normalize_to_max_one(df['extr_error'])
    df['val_normed'] = normalize_to_max_one(df['val_error'])
    df['complexity_normed'] = normalize_to_max_one(df['complexity'], defensive=False)

    if use_extrapolation:
        print('Extrapolation data used.')
        df['score'] = np.sqrt(df['extr_normed'] ** 2 + df['val_normed'] ** 2)
    else:
        print('No extrapolation data used, performing model selection based on complexity and validation instead.')
        df['score'] = np.sqrt(df['complexity_normed'] ** 2 + df['val_normed'] ** 2)

    return df


def normalize_to_max_one(arr, defensive=True):
    """
    Routine that normalizes an array to a maximumum of one.
    :param arr: array to be normalized
    :param defensive: flag to determine if behavior is defensive (if all array elements are the same raise exception)
                      or not (if all array elements are the same return an array of same length filled with zeros)
    """
    if np.isclose(np.max(arr), np.min(arr)):
        if defensive:
            raise ValueError('All elements in array are the same, no normalization possible.')
        else:
            return np.zeros(len(arr))
    norm_arr = arr / (np.max(arr))
    return norm_arr 


def aggregate_csv_files_recursively(directory, filename):
    """ Returns a pandas DF that is a concatenation of csvs with given filename in given directory (recursively)."""
    return pd.concat(_df_from_csv_recursive_generator(directory, filename))


def _df_from_csv_recursive_generator(directory, filename):
    """ Returns a generator producing pandas DF for each csv with given filename in given directory (recursively)."""
    for root, dirs, files in walk(directory):
        if filename in files:
            yield pd.read_csv(path.join(root, filename))

######################## functions from GMi ################################
            

def list_from_npy(directory, filename):
    """ Returns a generator producing list for each npy file with given filename in given directory (recursively).
    
    Args:
        directory: path to directory, where npy files are stored (in subfolders), string
        filename: name of npy file, string
        
    Returns:
        list1: list of all npy files found in the subdirectories, list    
    """
    list_npy = []
    for root, dirs, files in walk(directory):
        if filename in files:
            #print('here: ', root)
            list_npy.append(np.load(path.join(root, filename), allow_pickle=True))
            
    return list_npy

def get_list_of_formula(working_directory, filename='formula_0.npy'):
    """Returns list of formulas as sympy expression found in a the subfolders of a given directory.
    
    Args:
        working_directory: path to directory, where npy files are stored (in subfolders), string
        filename: name of npy file, string
        id1: list of id tags sorted with ascending score, list
        
    Returns:
        formula_sort: list of sympy expression of the formulas found in the subdirectories - sorted, list   
    """
    
    all_formula = list_from_npy(working_directory, filename)
    
    #print(len(all_formula))
    
    formula_sort = []
    
    #print(len(all_formula))

    for i in range(len(all_formula)):
        #print(i)
        #print(all_formula[i])
        formula_sort.append(sympy.sympify(all_formula[i]))
    
    return formula_sort
    

def graph_from_formula(formula):
    
    """returns variable tree from formula expression.
    
    Args:
        formula: list of sympy expressions, list
        
    Retruns:
        graph: list of variable trees of the respected formula expression, Source object
    
    """
    
    graph = []
    
    for i in range(len(formula)):
        #formula = parse_expr(formula[i]) # if input is string
        if not (formula[i] == np.nan):
            graph_c = Source(sympy.dotprint(formula[i]))
        else:
            print('formula is NaN')
            graph_c = np.nan
        graph.append(graph_c)
            
    return graph

def EQL_df_creation(num_runs, path_results, num_h_layers, reg_percent, use_extrapolation=True, filename_formula='formula_0.npy'):
    
    """function to create pandas dataframe from csv file for EQL runs. Ordered by id (ascending).
    
    Args:
        num_runs: number of runs, int
        path_results: path to working directory, where the results are stored, string
        num_h_layers: number of hidden layers, list with len=len(num_runs) or singel value
        reg_percent: regularization strength, list with len=len(num_runs)
        filename_formula: filename of npy file with formula expression in respected folder, string
            - e.g.: 'formula_0.npy'
            
    Returns:
        df: pandas dataframe with all info from the runs of EQL runs. nan where something is missing
    
    df includes:
        - 'val_error':  prediction error with validation dataset
        - 'extr_error': prediction error with test dataset
        - 'complexity': number of active nodes of network
        - 'id': the results are stored in a folder with this name. Serves as a tag
        - 'extr_normed': extr_error/max(extr_error) --> normed value for extr_error
        - 'val_normed': val_error/max(val_error) --> normed value for val_error
        - 'complexity_normed': complexity/max(complexity) --> normed value for complexity
        - 'score': root mean squared of extr_normed and val_normed
        - 'num_h_layers': number of hidden layers, int 
        - 'reg_percent': regularization strength, int 
        - 'formula': sympy formula expression resulting from EQL run
        - 'graph': variable tree of the respected formula expression, Source object
    """
    
    aggregated_results = aggregate_csv_files_recursively(path_results, "results.csv")
    
    df = select_instance(df=aggregated_results, use_extrapolation=use_extrapolation)
    
    # append df with nan, where no results are generated
    
    df_nan = pd.DataFrame({'Unnamed: 0': [np.nan], 
                'val_error': [np.nan],
                  'complexity': [np.nan],
                  'extr_error': [np.nan],
                  'id': [np.nan],
                  'extr_normed': [np.nan],
                  'val_normed': [np.nan],
                  'complexity_normed': [np.nan],
                  'score': [np.nan],
                  'formula': [np.nan],
                  }) 

    for i in range(num_runs):
        if any(df['id'] == i) == False: 
            df_nan['id'] = i
            df = df.append(df_nan, ignore_index=True)
    
    formula = get_list_of_formula(path_results, filename_formula)
    
    if not(len(formula) == num_runs):
        print('ERROR: # formulas in df are not equal to the # runs')
    
    
    # these lines need to be adapted for other input to be saved additionally
    # list_train_error = list_from_npy(path_results, 'loss_test_all.npy')
    # list_val_error = list_from_npy(path_results, 'loss_train_all.npy')
    # list_extr_error = list_from_npy(path_results, 'loss_val_all.npy')
    # list_complexity = list_from_npy(path_results, 'complexity_all.npy')
    
    # list_train_error = []
    # list_val_error = []
    # list_extr_error = []
    # list_complexity = []
    
    # for i in range(num_runs):
    #     list_train_error.append(np.load(path_results+'\\'+str(i)+'\\loss_train_all.npy'))
    #     list_val_error.append(np.load(path_results+'\\'+str(i)+'\\loss_val_all.npy'))
    #     list_extr_error.append(np.load(path_results+'\\'+str(i)+'\\loss_test_all.npy'))
    #     list_complexity.append(np.load(path_results+'\\'+str(i)+'\\complexity_all.npy'))
    
    
    df['num_h_layers'] = num_h_layers
    df['reg_percent'] = reg_percent
    df['formula'] = formula

    df.to_csv(path_or_buf=path_results+'\\df.csv')
    
    return df


def plot_from_df_reduced(df, title, x, y, details, logx=True, logy=True):
    
    """Function for plotting the results from the EQL.
    Only works in Jupyter Notebook!
    The plot shows x and y of EQL runs with different parameters:
    
    available details that can be displayed for the EQL model selection:
        - 'val_error':  prediction error with validation dataset
        - 'extr_error': prediction error with test dataset
        - 'complexity': number of active nodes of network
        - 'id': the results are stored in a folder with this name. Serves as a tag
        - 'extr_normed': extr_error/max(extr_error) --> normed value for extr_error
        - 'val_normed': val_error/max(val_error) --> normed value for val_error
        - 'complexity_normed': complexity/max(complexity) --> normed value for complexity
        - 'score': root mean squared of extr_normed and val_normed
        - 'num_h_layers': number of hidden layers (1,2,3,4), int 
        - 'reg_percent': regularization strength, int 
        - 'formula': sympy formula expression resulting from EQL run 
    
    Args:
        df: pandas dataframe with all input
            - for EQL model selection: expects all details mentioned above
        fun_name: name of the function - displayed in title, string
        x: data for x axis - must be inside df, string
        y: data for y axis - must be inside df, string
        details: details that should be displayed while hoovering over a point, list of strings
        logx: boolean if x-axis should be plotted logarithmic, default=True
        logy: boolean if y-axis should be plotted logarithmic, default=True
    
    E.g.: plot_extr_val(df_F1, 'F1', 'extr_error', 'val_error', ['complexity', 'id', 'formula'])
    """
    
    plot = hv.Scatter(df, [x, y], details)
    
    plot.opts(size=4.5, tools=['tap', 'hover'])
    
    plot.opts(logx=logx, logy=logy)
    plot.opts(title=title)
    plot.opts(xlabel=x, ylabel=y)
    plot.opts(width=700, height=700)
    plot.opts(legend_position='top_right')
    plot.opts(fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12})
    
    figure_bokeh = hv.render(plot, backend='bokeh')
    
    show(figure_bokeh)


if __name__ == '__main__':
    
    #path_data =                               # add path to directory, where the results are stored
    #num_h_layers =                            # number of hidden layers, list with len=len(num_runs) or singel value
    #rp =                                      # regularization strength, list with len=len(num_runs)

####### if file is missing --> generate npy file with the desired name and add them to the subdirectories to avoid error ########   

    #np.save(path_data+r'\formula_0.npy', [np.nan])
        
    #np.save(path_data+r'\loss_test_all.npy', [np.nan])
    #np.save(path_data+r'\loss_train_all.npy', [np.nan])
    #np.save(path_data+r'\loss_val_all.npy', [np.nan])
    #np.save(path_data+r'\complexity_all.npy', [np.nan])
        
    df = EQL_df_creation(num_runs=len(rp), path_results=path_data, num_h_layers=num_h_layers, reg_percent=rp, use_extrapolation=False, filename_formula='formula_0.npy')
    
    
    
    
    
    
    