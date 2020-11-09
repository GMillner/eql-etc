""" Neural Network Estimator for EQL - Equation Learner """
import math
import sys
from collections import namedtuple

import tensorflow as tf

import EQL_Layer_tf as eql
from data_utils import get_input_fns, extract_metadata
from evaluation import set_evaluation_hook
from utils import step_to_epochs, get_run_config, save_results, update_runtime_params_nomlflow, \
    get_div_thresh_fn, get_max_episode

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab

from timeout import time_limit, TimeoutException
from evaluation import calculate_complexity
#from evaluation import num_of_weights

import time
import os

# more network parameters are loaded from utils.py

default_params = {'model_base_dir': r'D:\GMi\Arbeitsordner_GMi\SymbolicRegression\JupyterNotebook\EQL_Tensorflow-master\_Git_repository_ECML\eql\results\nomlflow\F2',
                   'id': 2,  # job_id to identify jobs in result metrics file, separate model_dir for each id
                   'train_val_file': r'D:\GMi\Arbeitsordner_GMi\SymbolicRegression\JupyterNotebook\EQL_Tensorflow-master\_Git_repository_ECML\eql\example_data\F2data_train_val',  # Datafile containing training, validation data
                   'test_file': r'D:\GMi\Arbeitsordner_GMi\SymbolicRegression\JupyterNotebook\EQL_Tensorflow-master\_Git_repository_ECML\eql\example_data\F2data_test',  # Datafile containing test data, if set to None no test data is used
                   'epoch_factor': 1000,  # max_epochs = epoch_factor * num_h_layers
                   'num_h_layers': 2,  # number of hidden layers used in network
                   'generate_symbolic_expr': True,  # saves final network as a latex png and symbolic graph
                   'kill_summaries': True,  # reduces data generation, recommended when creating many jobs
                   'monitor_complexity': True,
                   #'L_reg': 1, # used exponentiell of the element-wise norm of the weight matrix used in regularization. For L2 = 2, L1 = 1
                   }

#default_params = {'model_base_dir': 'results\\_pre_train\\F1\\',
#                   'id': 24,  # job_id to identify jobs in result metrics file, separate model_dir for each id
#                   #'train_val_file': 'example_data/F1data_train_val',  # Datafile containing training, validation data
#                   #'test_file': 'example_data/F1data_test',  # Datafile containing test data, if set to None no test data is used
#                   'train_val_file': 'example_data/F1data_train_val',  # Datafile containing training, validation data
#                   'test_file': 'example_data/F1data_test',  # Datafile containing test data, if set to None no test data is used             
#                   'epoch_factor': 1000,  # max_epochs = epoch_factor * num_h_layers
#                   'num_h_layers': 1,  # number of hidden layers used in network
#                   'generate_symbolic_expr': True,  # saves final network as a latex png and symbolic graph
#                   'kill_summaries': True,  # reduces data generation, recommended when creating many jobs
#                  'monitor_complexity': False,    
#                   }


class Model(object):
    """ Class that defines a graph for EQL. """

    def __init__(self, mode, layer_width, num_h_layers, reg_start, reg_end, output_bound, weight_init_param, epoch_factor,
                 batch_size, test_div_threshold, reg_scale, l0_threshold, train_val_split, penalty_every, network_init_seed=None, **_):
        
        self.train_data_size = int(train_val_split * metadata['train_val_examples'])
        self.width = layer_width
        self.num_h_layers = num_h_layers
        self.weight_init_scale = weight_init_param / math.sqrt(metadata['num_inputs'] + num_h_layers)
        self.seed = network_init_seed
        self.reg_start = reg_start * (penalty_every+1)#math.floor(num_h_layers * epoch_factor * reg_sched[0])
        self.reg_end = reg_end * (penalty_every+1)#math.floor(num_h_layers * epoch_factor * reg_sched[1])
        self.output_bound = output_bound or metadata['extracted_output_bound']
        self.reg_scale = reg_scale
        self.batch_size = batch_size
        self.l0_threshold = l0_threshold
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        div_thresh_fn = get_div_thresh_fn(self.is_training, self.batch_size, test_div_threshold,
                                          train_examples=self.train_data_size)
        reg_div = namedtuple('reg_div', ['repeats', 'div_thresh_fn'])
        # functions defined in EQL --> see EQL_Layer-tf - Dict of fct 
        self.eql_layers = [eql.EQL_Layer(multiply=self.width, id=self.width, sub=self.width, exp=self.width, sin=self.width, cos=self.width,
                                         weight_init_scale=self.weight_init_scale, seed=self.seed)
                           for _ in range(self.num_h_layers)]
        #original
#        self.eql_layers = [eql.EQL_Layer(sin=self.width, cos=self.width, multiply=self.width, id=self.width,
#                                         weight_init_scale=self.weight_init_scale, seed=self.seed)
#                          for _ in range(self.num_h_layers)]
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
        normal_loss = tf.losses.get_total_loss() + P_theta 
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

@time_limit(10)
def evaluate_with_timeout(val_input, symbolic_hook):
    eqlearner.evaluate(input_fn=val_input, hooks=[symbolic_hook])

def calc_complexity_only(runtime_params, symbolic_hook, val_input):
    
    model = Model(mode= tf.estimator.ModeKeys.EVAL, **runtime_params)
    fns_list = [layer.get_fns() for layer in model.eql_layers]
    tresh = runtime_params['complexity_threshold']
    
    for i in range(1):
        try:
            evaluate_with_timeout(val_input, symbolic_hook)
        except TimeoutException or RecursionError:
            continue
        #print('here')    
    kernels = [value for key, value in symbolic_hook.weights.items() if 'kernel' in key.lower()]
    biases = [value for key, value in symbolic_hook.weights.items() if 'bias' in key.lower()]
    
    complexity = calculate_complexity(kernels, biases, fns_list, tresh)
    
    return complexity

# solves graphviz issue, where path of Graphviz executables are not found
#import os
os.environ["PATH"] += os.pathsep + r'D:\GMi\condaenv\GMi_EQL\Library\bin\graphviz'

#import time

## plotting function

def plot_error_conv(dict_loss, num_h_layers, reg_scale, monitor_complexity, reg_start, reg_end):
    
    """Function to plot error convergence.
    
    Args:
        dict_loss: dictonary with
            loss_train: list of train_error over training episodes, list
            loss_val: list of val_error over training episodes, list
            loss_test: list of extr_error over training episodes, list
            loss_compl: list of complexity over training episodes, list - optional
        num_h_layers: num_h_layers in run, scalar
        monitor_complexity: boolean if complexity is included or not, boolean
        reg_sched: begin and end of regularisation, (reg_start, reg_end) 
        
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
    
    #reg_start = reg_sched[0] * num_train_episodes
    #reg_end = reg_start + 10 #reg_sched[1] * num_train_episodes

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
    reg_scale_all = []
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # original:
    #runtime_params = update_runtime_params(sys.argv, default_params)

    runtime_params = update_runtime_params_nomlflow(sys.argv, default_params)
    
    metadata = extract_metadata(runtime_params['train_val_file'], runtime_params['test_file'])
    run_config = get_run_config(runtime_params['kill_summaries'])

    #eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
    #                                   params=runtime_params)

    logging_hook = tf.train.LoggingTensorHook(tensors={'train_accuracy': 'train_accuracy'}, every_n_iter=1000)
    evaluation_hook = set_evaluation_hook(**runtime_params)
    
    train_input, penalty_train_input, val_input, test_input = get_input_fns(**runtime_params, **metadata)
    
    max_episode = get_max_episode(**runtime_params)
    
    reg_duration = 6 # number of training episodes regularization is switched on
    reg_scale_0 = runtime_params['reg_scale']
    runtime_params['reg_scale'] = 0
    runtime_params['reg_start'] = runtime_params['reg_start'] * runtime_params['num_h_layers']
    runtime_params['reg_end'] = max_episode - 1 # maximum of training episodes, regularization is switched off
    
    eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
                                       params=runtime_params)
    
    
    print('One train episode equals %d normal epochs and 1 penalty epoch.' % runtime_params['penalty_every'])
    for train_episode in range(1, max_episode*4):
        
        print('Train episode: %d out of %d.' % (train_episode, max_episode))
        
        #runtime_params['reg_scale'] = reg_scale
        #mlflow.log_param('reg_scale', runtime_params['reg_scale'])
        #mlflow.log_metric('reg_scale', runtime_params['reg_scale'])
        
        # call eqlearner again to use updated params: reg_end & reg_scale
        eqlearner = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=runtime_params['model_dir'],
                                   params=runtime_params)
        
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! reg_scale: '+str(runtime_params['reg_scale']))
        
        penalty_flag = True
        eqlearner.train(input_fn=penalty_train_input)
        penalty_flag = False
        eqlearner.train(input_fn=train_input, hooks=[logging_hook])
        
        extr_results_train = eqlearner.evaluate(input_fn=train_input)
        extr_results_test = eqlearner.evaluate(input_fn=test_input)
        extr_results_val = eqlearner.evaluate(input_fn=val_input)
        
        loss_train_all.append(extr_results_train['loss'])
        loss_test_all.append(extr_results_test['loss'])
        loss_val_all.append(extr_results_val['loss'])
        if runtime_params['monitor_complexity']: 
            extr_complexity = calc_complexity_only(runtime_params, evaluation_hook, val_input)
            print('complexity: ', extr_complexity)
            complexity_all.append(extr_complexity)
            
            if runtime_params['reg_end'] > train_episode >= runtime_params['reg_start'] + reg_duration:
                conv_compl_reg = sum(abs(np.diff(complexity_all[-3:], n=1)))
                
                if conv_compl_reg == 0:
                    runtime_params['reg_end'] = train_episode
        else:
            conv_compl = 0
                
        if (runtime_params['reg_start'] + reg_duration) > train_episode >= runtime_params['reg_start']:
            runtime_params['reg_scale'] = runtime_params['reg_scale'] + reg_scale_0 * (1/reg_duration)
        if train_episode ==  runtime_params['reg_end']:
            runtime_params['reg_scale'] = 0
        reg_scale_all.append(runtime_params['reg_scale'])
#        error_convergence = False
#        
#        if train_episode >= max_episode:
#            loss_conv_train = abs(loss_train_all[-1]-loss_train_all[-2]) + abs(loss_train_all[-1]-loss_train_all[-5])
#            loss_conv_val = abs(loss_val_all[-1]-loss_val_all[-2]) + abs(loss_val_all[-1]-loss_val_all[-5])
#            conv_complexity = abs(complexity_all[-1]-complexity_all[-2]) + abs(complexity_all[-1]-complexity_all[-3])
#            
#            if loss_conv_train <= loss_train_all[-1]*0.1 and loss_conv_val <= loss_val_all[-1]*0.1 and conv_complexity == 0:
#                error_convergence = True
#                
#        if error_convergence == True:
#            print('!!!!!!!!!!!!!!!!!!!!!Error is converging at training episode: '+str(train_episode))
#            break        
        
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
    np.save(os.path.join(runtime_params['model_dir'], 'loss_test_all.npy'), loss_test_all)
    np.save(os.path.join(runtime_params['model_dir'], 'reg_scale_all.npy'), reg_scale_all)
    
    if runtime_params['monitor_complexity']: 
        np.save(os.path.join(runtime_params['model_dir'], 'complexity_all.npy'), complexity_all)
        
    
    val_results = eqlearner.evaluate(input_fn=val_input, name='validation', hooks=[evaluation_hook])
    results = dict(val_error=val_results['loss'], complexity=evaluation_hook.get_complexity())
    if test_input is not None:  # test_input function is only provided if extrapolation data is given
        extr_results = eqlearner.evaluate(input_fn=test_input, name='extrapolation')
        results['extr_error'] = extr_results['loss']
    save_results(results, runtime_params)
    print('Model evaluated. Results:\n', results)
    
    end = time.time()
    print('Time (s): '+str(end - start))
    
    # log model: https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html

    #mlflow.tensorflow.log_model(runtime_params['model_dir'], )
    #mlflow.tensorflow.save_model(runtime_params['model_dir'], )
    
    ######## create plots ###########
    
    dict_loss = {'loss_test': loss_test_all,
           'loss_train': loss_train_all,
           'loss_val': loss_val_all,
           'loss_compl': complexity_all,
           }
    
    plot_error_conv(dict_loss, runtime_params['num_h_layers'], reg_scale_0, runtime_params['monitor_complexity'], runtime_params['reg_start'], runtime_params['reg_end'])
    plt.savefig((os.path.join(runtime_params['model_dir'],'error_convergence.png')))

#old
#    plt.figure(figsize=(15,15))
#    
#    plt.plot(loss_train_all, 'b--*', label='training error')
#    #plt.plot(loss_test_all, label='test error')
#    #plt.plot(loss_val_all, 'r--*', label='validation error')
#    #plt.plot(complexity_all, 'k--*', label='complexity')
#    plt.xlabel('# Training episodes', size=50)
#    plt.ylabel('Loss', size=50)
#
#    plt.legend(prop={'size': 30})
#    plt.savefig(os.path.join(runtime_params['model_dir'],'Loss_over_episodes_training.png'))
#    
#    plt.figure(figsize=(15,15))
#
#    plt.plot(loss_train_all, 'b--*', label='training error')
#    plt.plot(loss_test_all, 'g--*', label='test error')
#    #plt.plot(loss_val_all, 'r--*', label='validation error')
#    plt.xlabel('# Training episodes', size=50)
#    plt.ylabel('Loss', size=50)
#
#    plt.legend(prop={'size': 30})
#
#    #plt.ylim([0,3])
#    
#    plt.savefig(os.path.join(runtime_params['model_dir'],'Loss_over_episodes_test.png'))
#    
#    plt.figure(figsize=(15,15))
#
#    plt.plot(loss_train_all, 'b--*', label='training error')
#    plt.plot(loss_test_all, 'g--*', label='test error')
#    #plt.plot(loss_val_all, 'r--*', label='validation error')
#    plt.xlabel('# Training episodes', size=50)
#    plt.ylabel('Loss', size=50)
#
#    plt.legend(prop={'size': 30})
#
#    plt.ylim([0,1])
#    
#    plt.savefig(os.path.join(runtime_params['model_dir'],'Loss_over_episodes_test-cut.png'))
#
#    if runtime_params['monitor_complexity']: 
#        plt.figure(figsize=(15,15))
#    
#        plt.plot(complexity_all, 'k--*', label='complexity')
#        plt.xlabel('# Training episodes', size=50)
#        plt.ylabel('# active nodes in graph', size=50)
#    
#        plt.legend(prop={'size': 20})
#        
#        plt.savefig(os.path.join(runtime_params['model_dir'],'Complexity_over_episodes.png'))