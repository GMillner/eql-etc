"""
Created on Thu Nov  7 09:39:10 2019

@author: g.millner

Python file to show the results of the model selection (EQL).
Opens web browser with interactive graph. 

call bokeh_server.stop() for closing the Bokeh server.

"""

__author__ = "Gerfried Millner (GMi)" 
__version__ = "1.0.0" 
__date__ = "09.11.2020"
__email__ = "g.millner@gmx.at" 
__status__ = "Development" 


#import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#import bokeh
#from bokeh.io import show
import pandas as pd
import panel as pn
#import os

#import sympy
#import pickle

import holoviews as hv
#from holoviews import opts
hv.extension('bokeh','matplotlib')


def plot_extr_val_py(df, fun_name, details):
    
    """Function for plotting the results from the model selection routine [EQL].
    --> with workaround to work outside a jupyter notebook. Opens web browser with graph.
    
    The plot shows extrapolation and validation error of EQL runs with different parameters:
    number of hidden layer (num_h_layer) and regularization strength of neural network (reg_scale)
    are variied. The values are ordered with ascending score.
    
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
        - 'reg_scale': regularization strength, int 
        - 'formula': sympy formula expression resulting from EQL run
    
    Args:
        df: pandas dataframe with all input
            - for EQL model selection: expects all details mentioned above
        fun_name: name of the function - displayed in title, string
        details: details that should be displayed while hoovering over a point, list of strings
    
    E.g.: plot_extr_val(df_F1, 'F1', ['complexity', 'id', 'formula'])
    """

    index_1 = [i for i, n in enumerate(df['num_h_layers']) if n == 1]
    index_2 = [i for i, n in enumerate(df['num_h_layers']) if n == 2]
    index_3 = [i for i, n in enumerate(df['num_h_layers']) if n == 3]
    index_4 = [i for i, n in enumerate(df['num_h_layers']) if n == 4]
    
    plot_1m = hv.Scatter(df, ['extr_normed', 'val_normed'], details, label='num_h_layers = 1')
    plot_1 = plot_1m.iloc[index_1].opts(size=3.5, color='b', tools=['tap', 'hover'])
    plot_2m = hv.Scatter(df, ['extr_normed', 'val_normed'], details, label='num_h_layers = 2')
    plot_2 = plot_2m.iloc[index_2].opts(size=3.5, color='r', tools=['tap', 'hover'])
    plot_3m = hv.Scatter(df, ['extr_normed', 'val_normed'], details, label='num_h_layers = 3')
    plot_3 = plot_3m.iloc[index_3].opts(size=3.5, color='g', tools=['tap', 'hover'])
    plot_4m = hv.Scatter(df, ['extr_normed', 'val_normed'], details, label='num_h_layers = 4')
    plot_4 = plot_4m.iloc[index_4].opts(size=3.5, color='m', tools=['tap', 'hover'])
    plot_best_score1 = hv.Scatter(df, ['extr_normed', 'val_normed'], details, label='best score')

    color_best_score_all = ['b','r','g','m']
    cbs = color_best_score_all[df['num_h_layers'][0]-1]

    plot_best_score = plot_best_score1.iloc[0].opts(size=10, color=cbs, marker='*', tools=['tap', 'hover'])

    plot = plot_1 * plot_2 * plot_3 * plot_4 * plot_best_score
    
    #plot_list = [plot_1, plot_2, plot_3, plot_4, plot_best_score]

    plot.opts(logx=True, logy=True)
    plot.opts(title="Model selection results for "+fun_name)
    plot.opts(xlabel='extrapolation error - normed', ylabel='validation error - normed')
    plot.opts(width=900, height=900)
    plot.opts(legend_position='top_right')
    plot.opts(fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12})
    
    #figure_bokeh = hv.render(plot, backend='bokeh')
    #show(figure_bokeh)
    
    #bokeh_server = pn.Row(plot).show(port=12345)
    
    return plot

def plot_from_df(df, fun_name, x, y, details, logx=True, logy=True):
    
    """Function for plotting the results from the model selection routine [EQL].
    Only works in Jupyter Notebook!
    The plot shows x and y of EQL runs with different parameters:
    number of hidden layer (num_h_layer) and regularization strength of neural network (reg_scale)
    are variied. The values are ordered with ascending score.
    
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
        - 'reg_scale': regularization strength, int 
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
    
    index_1 = [i for i, n in enumerate(df['num_h_layers']) if n == 1]
    index_2 = [i for i, n in enumerate(df['num_h_layers']) if n == 2]
    index_3 = [i for i, n in enumerate(df['num_h_layers']) if n == 3]
    index_4 = [i for i, n in enumerate(df['num_h_layers']) if n == 4]    
    
    plot_1m = hv.Scatter(df, [x, y], details, label='num_h_layers = 1')
    plot_1 = plot_1m.iloc[index_1].opts(size=3.5, color='b', tools=['tap', 'hover'])
    plot_2m = hv.Scatter(df, [x, y], details, label='num_h_layers = 2')
    plot_2 = plot_2m.iloc[index_2].opts(size=3.5, color='r', tools=['tap', 'hover'])
    plot_3m = hv.Scatter(df, [x, y], details, label='num_h_layers = 3')
    plot_3 = plot_3m.iloc[index_3].opts(size=3.5, color='g', tools=['tap', 'hover'])
    plot_4m = hv.Scatter(df,[x, y], details, label='num_h_layers = 4')
    plot_4 = plot_4m.iloc[index_4].opts(size=3.5, color='m', tools=['tap', 'hover'])
    plot_best_score1 = hv.Scatter(df,[x, y], details, label='best score')

    color_best_score_all = ['b','r','g','m']
    cbs = color_best_score_all[df['num_h_layers'][0]-1]

    plot_best_score = plot_best_score1.iloc[0].opts(size=10, color=cbs, marker='*', tools=['tap', 'hover'])

    plot = plot_1 * plot_2 * plot_3 * plot_4 * plot_best_score

    plot.opts(logx=logx, logy=logy)
    plot.opts(title="Model selection results for "+fun_name)
    plot.opts(xlabel=x, ylabel=y)
    plot.opts(width=700, height=700)
    plot.opts(legend_position='top_right')
    plot.opts(fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12})

    #figure_bokeh = hv.render(plot, backend='bokeh')
    #show(figure_bokeh)
    
    #bokeh_server = pn.Row(plot).show(port=12345)
    
    return plot


if __name__ == '__main__':
    
    
    # example | plot results from model selection results of F1 and F2 
    
    path_to_model_selection = r''       # path do the directory of the model selection
    dataframe_store = r''               # directory where DataFrames are stored
    df_F1 = pd.read_csv(dataframe_store+'\\df_F1.csv')
    df_F2 = pd.read_csv(dataframe_store+'\\df_F2.csv')
    
    extr_val_F_1 = plot_extr_val_py(df_F1, 'F1', ['score', 'reg_scale', 'complexity', 'id', 'formula'])
    extr_val_F_2 = plot_extr_val_py(df_F2, 'F2', ['score', 'reg_scale', 'complexity', 'id', 'formula'])

    plot_F1 = plot_from_df(df_F1, 'F1', 'val_error', 'complexity', ['score', 'reg_scale', 'id', 'formula'], logx=True, logy=True)
    plot_F2 = plot_from_df(df_F2, 'F2', 'val_error', 'complexity', ['score', 'reg_scale', 'id', 'formula'], logx=True, logy=True)

    bokeh_server = pn.Row(extr_val_F_1).show(port=12345) 
    
    # stop the bokeh server (when needed)
    #bokeh_server.stop()
    




