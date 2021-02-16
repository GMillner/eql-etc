""" Neural Network Estimator for EQL - Equation Learner """
import math
import os
import sys
import time
from collections import namedtuple

import tensorflow as tf

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

import mlflow
import mlflow.tensorflow
import argparse

#mlflow.set_tracking_uri('')

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.join(file_path, '..', '..')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from src import config
from src.timeout import time_limit, TimeoutException
from src.evaluation import calculate_complexity
#from evaluation import num_of_weights

import src.EQL_Layer_tf as eql
from src.data_utils import get_input_fns, extract_metadata
from src.evaluation import set_evaluation_hook, save_symbolic_expression
from src.utils import step_to_epochs, get_run_config, save_results, update_runtime_params, \
    get_div_thresh_fn, get_max_episode

import logging
#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
 
# cpu only mode
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

__author__ = "Gerfried Millner (GMi)"
__version__ = "0.2.4"
__date__ = "16.02.2021"
__email__ = "g.millner@gmx.at" 
__status__ = "Development"

# create a file and a console handler
#fileHandler = logging.FileHandler(os.path.join(EXEC_PATH, 'eqlearner.log'))
fileHandler = logging.FileHandler(os.path.join(lib_path, 'eqlearner.log'))
fileHandler.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
 
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(threadName)-12.12s] - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
consoleHandler.setFormatter(formatter)
 
# add the file handler to the logger
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

# parameters are loaded from config.py

class Model(object):
    """ Class that defines a graph for EQL. """

    def __init__(self, mode, layer_width, num_h_layers, reg_start, reg_end, output_bound, weight_init_param, epoch_factor,
                 batch_size, test_div_threshold, reg_scale, penalty_every, l0_threshold, train_val_split, network_init_seed=None, **_):
        
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.width = layer_width
        self.num_h_layers = num_h_layers
        self.weight_init_scale = weight_init_param / math.sqrt(metadata['num_inputs'] + num_h_layers)
        self.seed = network_init_seed
        self.reg_start = reg_start  * (penalty_every+1)
        self.reg_end = reg_end  * (penalty_every+1)
        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size, test_div_threshold,
                                          train_examples=self.train_data_size)
        
        reg_div = namedtuple('reg_div', ['repeats', 'div_thresh_fn'])
        # functions defined in EQL --> see EQL_Layer-tf - Dict of fct 

        self.eql_layers = [eql.EQL_Layer(sin=self.width, cos=self.width, multiply=self.width, id=self.width,
                                        weight_init_scale=self.weight_init_scale, seed=self.seed)
                          for _ in range(self.num_h_layers)]
        # regularized division:
        self.eql_layers.append(
            eql.EQL_Layer(reg_div=reg_div(repeats=metadata['num_outputs'], div_thresh_fn=div_thresh_fn),
                          weight_init_scale=self.weight_init_scale, seed=self.seed))
        

    def __call__(self, inputs):
        global_step = tf.train.get_or_create_global_step()
        num_epochs = step_to_epochs(global_step, self.batch_size, self.train_data_size)
        l1_reg_sched = tf.multiply(tf.cast(tf.less(num_epochs, self.reg_end), tf.float32),
                                   tf.cast(tf.greater(num_epochs, self.reg_start), tf.float32)) * self.reg_scale
        l0_threshold = tf.cond(tf.less(num_epochs, self.reg_end), lambda: tf.zeros(1), lambda: self.l0_threshold)

        output = inputs
        for layer in self.eql_layers:
            output = layer(output, l1_reg_sched=l1_reg_sched, l0_threshold=l0_threshold)

        P_bound = (tf.abs(output) - self.output_bound) * tf.cast((tf.abs(output) > self.output_bound), dtype=tf.float32)
        tf.add_to_collection('Bound_penalties', P_bound)
        return output


def model_fn(features, labels, mode, params):
    """ The model_fn argument for creating an Estimator. """
    model = Model(mode=mode, **params)
    evaluation_hook.init_network_structure(model, params)
    
    global_step = tf.train.get_or_create_global_step()
    input_data = features
    predictions = model(input_data)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum([tf.reduce_mean(reg_loss) for reg_loss in reg_losses], name='reg_loss_mean_sum')# L2 with **2
        bound_penalty = tf.reduce_sum(tf.get_collection('Bound_penalties'))
        P_theta = tf.reduce_sum(tf.get_collection('Threshold_penalties'))
        penalty_loss = P_theta + bound_penalty
        mse_loss = tf.losses.mean_squared_error(labels, predictions)
        #normal_loss = tf.losses.get_total_loss() + P_theta
        normal_loss = tf.losses.get_total_loss() + P_theta #+ sympy.diff(, x_1)
        loss = penalty_loss if penalty_flag else normal_loss
        train_accuracy = tf.identity(
            tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)[1], name='train_accuracy')
        tf.summary.scalar('total_loss', loss, family='losses')
        tf.summary.scalar('MSE_loss', mse_loss, family='losses')  # inaccurate for penalty epochs (ignore)
        tf.summary.scalar('Penalty_Loss', penalty_loss, family='losses')
        tf.summary.scalar("Regularization_loss", reg_loss, family='losses')
        tf.summary.scalar('train_acc', train_accuracy, family='accuracies')  # inaccurate for penalty epochs (ignore)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=tf.train.AdamOptimizer(params['learning_rate'], beta1=params['beta1']).minimize(loss, global_step))
            #train_op=tf.train.GradientDescentOptimizer(params['learning_rate']).minimize(loss, global_step))
    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.sqrt(tf.losses.mean_squared_error(labels, predictions))
        eval_acc_metric = tf.metrics.percentage_below(values=tf.abs(labels - predictions), threshold=0.02)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL, loss=loss,
                                          eval_metric_ops={'eval_accuracy': eval_acc_metric})

###### added by GMi #########
#from utils import *
#from functools import reduce        
#from timeout import time_limit, TimeoutException
#from evaluation import calculate_complexity0

# @time_limit(10)
# def evaluate_with_timeout(val_input, symbolic_hook):
#     eqlearner.evaluate(input_fn=val_input, hooks=[symbolic_hook])

# def calc_complexity_only(runtime_params, symbolic_hook, val_input):
    
#     model = Model(mode= tf.estimator.ModeKeys.EVAL, **runtime_params)
#     fns_list = [layer.get_fns() for layer in model.eql_layers]
#     tresh = runtime_params['complexity_threshold']
    
#     for i in range(1):
#         try:
#             evaluate_with_timeout(val_input, symbolic_hook)
#         except TimeoutException or RecursionError:
#             continue
#         #print('here')    
#     kernels = [value for key, value in symbolic_hook.weights.items() if 'kernel' in key.lower()]
#     biases = [value for key, value in symbolic_hook.weights.items() if 'bias' in key.lower()]
    
#     complexity = calculate_complexity(kernels, biases, fns_list, tresh)
    
#     return complexity


def sum_weights(kernel_list):
    
    sum_weights = []
    
    for i in range(len(kernel_list)):
        
        sum_weight_c = 0
        for j in range(len(kernel_list[0])):
            sum_weight_c = sum_weight_c + np.sum(abs(kernel_list[i][j]))
        
        sum_weights.append(sum_weight_c)
    
    return np.array(sum_weights)

## plotting function

def plot_error_conv(dict_loss, num_h_layers, reg_scale, monitor_complexity, reg_start, reg_end):
    
    """Function to plot error convergence.
    
    Args:
        dict_loss: dictonary with
            loss_train: list of train_error over training episodes, list
            loss_val: list of val_error over training episodes, list
            loss_test: list of extr_error over training episodes, list
            loss_compl: list of complexity over training episodes, list - optional
        num_h_layers: num_h_layers in run, int
        monitor_complexity: boolean if complexity is included or not, boolean
        reg_start: start of regularisation in training episodes, int
        reg_end: end of regularization in training episodes, int
        
    Returns:
    
    needs the following packages:
        import matplotlib.pylab as pylab
        from matplotlib import pyplot as plt
    """
    
    #import matplotlib.pylab as pylab --> pylab is mandatory to ensure ticksize
    
    params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 15),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
    pylab.rcParams.update(params)
    
    ########################################################
    
    train_error = dict_loss['loss_train']
    val_error = dict_loss['loss_val']
    test_error = dict_loss['loss_test']
    if monitor_complexity==True:
        complexity = dict_loss['loss_compl']

    #num_train_episodes = num_h_layers * 20
    #training_episodes = np.linspace(1,num_train_episodes-1,num_train_episodes-1)

    if monitor_complexity==True:
        fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'hspace': 0})
    else:
        fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})

    axs[0].semilogy(train_error, 'o-', markersize=10)

    axs[0].axvspan(int(reg_start-1), int(reg_end-1), color='yellow', alpha=0.5)
    axs[0].set_ylabel('Loss (train)', size=30)
    axs[0].grid()
    
    axs[1].semilogy(val_error, 'o-', markersize=10)

    axs[1].axvspan(int(reg_start-1), int(reg_end-1), color='yellow', alpha=0.5)
    axs[1].set_ylabel('Loss (val)', size=30)
    axs[1].grid()

    axs[2].semilogy(test_error, 'o-', markersize=10)

    axs[2].axvspan(int(reg_start-1), int(reg_end-1), color='yellow', alpha=0.5)
    axs[2].set_ylabel('Loss (test)', size=30)
    axs[2].grid()
    
    if monitor_complexity==True:
        axs[3].plot(complexity, 'o-', markersize=10)

        axs[3].axvspan(int(reg_start-1), int(reg_end-1), color='yellow', alpha=0.5)
        axs[3].set_ylabel('# active nodes', size=30)
        axs[3].grid()

        axs[3].set_xlabel('# training episodes', size=30)
    #plt.title('num_h_layers = '+str(num_h_layers)+' - reg_scale: 0...'+str(reg_scale))
    
###############################################################################

if __name__ == '__main__':

    start = time.time()
    loss_test_all = []
    loss_train_all = []
    loss_val_all = []
    complexity_all = []
    L1_all = []
    reg_scale_all = []
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    runtime_params = update_runtime_params(sys.argv, config.default_params)
    
    metadata = extract_metadata(runtime_params['train_val_file'], runtime_params['test_file'])
    run_config = get_run_config(runtime_params['kill_summaries'])
    
    with mlflow.start_run(run_name=runtime_params['model_dir']):
        
        mlflow.tensorflow.autolog()
    
        logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
        evaluation_hook = set_evaluation_hook(**runtime_params)
        
        train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)
        
        max_episode = get_max_episode(**runtime_params)
        
        runtime_params['reg_scale'] = 0
        runtime_params['reg_start'] = int(runtime_params['reg_sched'][0] * max_episode)
        runtime_params['reg_end'] =  int(runtime_params['reg_sched'][1] * max_episode)
        
        eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
                                           params=runtime_params)
         
        # MLFlow parameter logging
        
        mlflow.log_param('model_base_dir', runtime_params['model_base_dir'])
        mlflow.log_param('id', runtime_params['id'])
        mlflow.log_param('train_val_file', runtime_params['train_val_file'])
        mlflow.log_param('test_file', runtime_params['test_file'])
        mlflow.log_param('epoch_factor', runtime_params['epoch_factor'])
        mlflow.log_param('num_h_layers', runtime_params['num_h_layers'])
        mlflow.log_param('generate_symbolic_expr', runtime_params['generate_symbolic_expr'])
        mlflow.log_param('kill_summaries', runtime_params['kill_summaries'])
        mlflow.log_param('monitor_complexity', runtime_params['monitor_complexity'])
        #mlflow.log_param('L_reg', runtime_params['L_reg'])
        mlflow.log_param('train_val_split', runtime_params['train_val_split'])
        mlflow.log_param('layer_width', runtime_params['layer_width'])
        mlflow.log_param('batch_size', runtime_params['batch_size'])
        mlflow.log_param('learning_rate', runtime_params['learning_rate'])
        mlflow.log_param('l0_threshold', runtime_params['l0_threshold'])
        #mlflow.log_param('reg_scale', reg_scale_0)
        mlflow.log_param('reg_percent', runtime_params['reg_percent'])
        mlflow.log_param('reg_start', runtime_params['reg_start'])
        mlflow.log_param('output_bound', runtime_params['output_bound'])
        mlflow.log_param('weight_init_param', runtime_params['weight_init_param'])
        mlflow.log_param('test_div_threshold', runtime_params['test_div_threshold'])
        mlflow.log_param('complexity_threshold', runtime_params['complexity_threshold'])
        mlflow.log_param('penalty_every', runtime_params['penalty_every'])
        mlflow.log_param('penalty_bounds', runtime_params['penalty_bounds'])
        mlflow.log_param('network_init_seed', runtime_params['network_init_seed'])
        
        print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
        for train_episode in range(1, max_episode*4):
            
            print('Train episode: %d out of %d.' % (train_episode, max_episode))
            
            
            # call eqlearner again to use updated params: reg_end & reg_scale
            eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
                                       params=runtime_params)
            
            
            penalty_flag = True
            eqlearner.train(input_fn=penalty_train_input)
            penalty_flag = False
            eqlearner.train(input_fn=train_input, hooks=[logging_hook])
            
            extr_results_train = eqlearner.evaluate(input_fn=train_input)
            
            if test_input is not None:
                extr_results_test = eqlearner.evaluate(input_fn=test_input)
            extr_results_val = eqlearner.evaluate(input_fn=val_input)
            
            if os.path.isfile(os.path.join(runtime_params['model_dir'],'kernels.npy')):
                kernels = np.load(os.path.join(runtime_params['model_dir'],'kernels.npy'), allow_pickle=True)
                L1 = np.sum(sum_weights(kernels))
                L1_all.append(L1)
                mlflow.log_metric("L1", L1)
    
            if train_episode == runtime_params['reg_start']:
                runtime_params['reg_scale']  = runtime_params['reg_percent'] * extr_results_train['loss'] / L1
                reg_scale_0 = runtime_params['reg_scale']
                reg_scale_all.append(reg_scale_0)
                mlflow.log_metric("reg_scale", runtime_params['reg_scale'])

                
            mlflow.log_metric("loss_train", extr_results_train['loss'])
            if test_input is not None:
                mlflow.log_metric("loss_test", extr_results_test['loss'])
            mlflow.log_metric("loss_val", extr_results_val['loss'])
            
            loss_train_all.append(extr_results_train['loss'])
            if test_input is not None:    
                loss_test_all.append(extr_results_test['loss'])
            loss_val_all.append(extr_results_val['loss'])
            if runtime_params['monitor_complexity']: 
                
                eqlearner.evaluate(input_fn=val_input, hooks=[evaluation_hook])
                extr_complexity = evaluation_hook.get_complexity()
                
                print('complexity: ', extr_complexity)
                complexity_all.append(extr_complexity)
                mlflow.log_metric("complexity", extr_complexity)
                
            else:
                conv_compl = 0
                    
            if train_episode ==  runtime_params['reg_end']:
                runtime_params['reg_scale'] = 0
            
            if train_episode >= max_episode:
                conv_train = sum(abs(np.diff(loss_train_all[-5:], n=1)))
                conv_val = sum(abs(np.diff(loss_val_all[-5:], n=1)))
                if runtime_params['monitor_complexity']:
                    conv_compl = sum(abs(np.diff(complexity_all[-5:], n=1)))
                
                if conv_train <= loss_train_all[-1]*0.1 and conv_train <= loss_val_all[-1]*0.1 and conv_compl == 0:
                    print('!!!!!!!!!!!!!!!!!!!!!!!Error converge at: '+str(train_episode))
                    break
                    
        print('Training complete. Evaluating...')
        
        np.save(os.path.join(runtime_params['model_dir'], 'loss_train_all.npy'), loss_train_all)
        np.save(os.path.join(runtime_params['model_dir'], 'loss_val_all.npy'), loss_val_all)
        if test_input is not None:
            np.save(os.path.join(runtime_params['model_dir'], 'loss_test_all.npy'), loss_test_all)
        
        if runtime_params['monitor_complexity']: 
            np.save(os.path.join(runtime_params['model_dir'], 'complexity_all.npy'), complexity_all)
            #mlflow.log_metric("complexity_all", complexity_all)

        np.save(os.path.join(runtime_params['model_dir'],'reg_scale.npy'), reg_scale_all)
        np.save(os.path.join(runtime_params['model_dir'],'L1_all.npy'), L1_all)
        
        val_results = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook])
        results = dict(val_error=val_results['loss'], complexity=evaluation_hook.get_complexity())
        if test_input is not None:  # test_input function is only provided if extrapolation data is given
            extr_results = eqlearner.evaluate(input_fn=test_input, name='extrapolation')
            results['extr_error'] = extr_results['loss']
        save_results(results, runtime_params)
        print('Model evaluated. Results:\n', results)
        
        end = time.time()
        print('Time (s): '+str(end - start))
        
        ######### create plots ###########
        
        dict_loss = {'loss_test': loss_test_all,
               'loss_train': loss_train_all,
               'loss_val': loss_val_all,
               'loss_compl': complexity_all,
               }
        
        plot_error_conv(dict_loss, runtime_params['num_h_layers'], reg_scale_0, runtime_params['monitor_complexity'], runtime_params['reg_start'], runtime_params['reg_end'])
        plt.savefig((os.path.join(runtime_params['model_dir'],'error_convergence.png')))
        
        ##### create formula #####

        if os.path.isfile(os.path.join(runtime_params['model_dir'],'kernels.npy')):
        
            model_formula = Model(mode= tf.estimator.ModeKeys.EVAL, **runtime_params)
            fns_list = [layer.get_fns() for layer in model_formula.eql_layers]
            
            kernels = np.load(os.path.join(runtime_params['model_dir'],'kernels.npy'), allow_pickle=True)
            biases = np.load(os.path.join(runtime_params['model_dir'],'biases.npy'), allow_pickle=True)
            
            save_symbolic_expression(kernels, biases, fns_list, runtime_params['model_dir'], round_decimals=3)
            
    

