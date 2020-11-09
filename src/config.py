import os
lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

__author__ = "Gerfried Millner"
__version__ = "0.2.0"
__date__ = "30.03.2020"
__email__ = "g.millner@gmx.at" 
__status__ = "Development"


default_params = {
   'model_base_dir': os.path.join(lib_path, 'models', 'F1_test'),
   'id': 2,  # job_id to identify jobs in result metrics file, separate model_dir for each id
   'train_val_file': lib_path + r'\data\example_data\F1data_train_val',  
   'test_file': None,
   'epoch_factor': 1000,  # max_epochs = epoch_factor * num_h_layers
   'num_h_layers': 1,  # number of hidden layers used in network
   'generate_symbolic_expr': True,  # saves final network as a latex png and symbolic graph
   'kill_summaries': True,  # reduces data generation, recommended when creating many jobs
   'monitor_complexity': True,
   'reg_percent': 0.01, # percentage of loss_train at reg_start
   'reg_start': 2, # starting point of the regularization in training episodes = penalty_every + 1 epochs
   }


# The following parameters should not be changed in most cases. 
network_parameters = {'train_val_split': .9,  # how data in train_val_file is split, .9 means 90% train 10% validation
                      'layer_width': 10,  # number of identical nodes per hidden layer - custom: 10
                      'batch_size': 20,  # size of data batches used for training - custom: 20
                      'learning_rate': 5e-4, # original
                      'beta1': .4,
                      'l0_threshold': .05,  # threshold for regularization, see paper: chapter 2.3 Reg Phases
                      'output_bound': None,  # output boundary for penalty epochs, if set to None it is calculated
                      # from training/validation data
                      'weight_init_param': 1.,
                      'test_div_threshold': 1e-4,  # threshold for denominator in division layer used when testing
                      'complexity_threshold': 0.01,  # determines how small a weight has to be to be considered inactive
                      'penalty_every': 50,  # feed in penalty data for training and evaluate after every n epochs
                      'penalty_bounds': None,  # domain boundaries for generating penalty data, if None it is calculated
                      # from extrapolation_data (if provided) or training/validation data
                      'network_init_seed': None,  # seed for initializing weights in network
                      }

