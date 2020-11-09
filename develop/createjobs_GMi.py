"""
Generation of job files for model selection.
    - *generate_jobs* creates shell script files for multiple jobs. Varying the number of hidden layers and other
    parameters allows for effective model selection. Also creates a submission file to submit all jobs.
"""
import os
#from ast import literal_eval
#from sys import argv

import numpy as np

#job_dir = 'jobs'
#result_dir = os.path.join("results", "model_selection")
#submitfile = os.path.join('jobs', 'submit.bat')

def generate_jobs_bat(train_val_file, test_file, anaconda_activate_dir, env_dir, EQL_dir, EQL_drive, input_fun):
    
    """ creates batch files for multiple jobs. Varying the number of hidden layers and other
    parameters allows for effective model selection. Also creates a submission file to submit all runs.
    When using a conda environment - otherwise: generate_jobs_bat_base
    
    Args:
        train_val_file: Datafile containing training, validation data, string
        test_file: Datafile containing test data, string
        anaconda_activate_dir: directory to activate.bat in conda environment, string
        env_dir: conda einvironment directory, string
        EQL_dir: master directory where the train.py file is: E.g. ...\EQL_Tensorflow-master, string 
        EQL_drive: drive where EQL_dir is (needs to be called when changing drives) - E.g. 'D:', string
        input_fun: name of the Data input - will be added to the jobs and results directory, string
    
    Returns: 
        creates .bat files in job_dir ('jobs/input_fun')
    """
    job_dir = os.path.join('jobs', input_fun)
    result_dir = os.path.join("results", input_fun)  
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)
    pwd = os.getcwd()
    id = 0
    l1_reg_range = 10 ** (-np.linspace(45, 60, num=5) / 10.0) # 10 values for reg_range
    cmd_all = []
    for l1_reg_scale in l1_reg_range:
        for num_h_layers in [1, 2, 3]: # num_h_layers
            params = dict(model_base_dir=result_dir, train_val_file=train_val_file, test_file=test_file, id=id,
                          num_h_layers=num_h_layers, reg_scale=l1_reg_scale, kill_summaries=True,
                          generate_symbolic_expr=True)
            dict_str = str(params)
            cmd = '{} "{}"'.format('call python ' + os.path.join(pwd, 'train.py '), dict_str) 
            cmd_all.append(cmd)
            script_fname = os.path.join(job_dir, str(id) + ".bat")
            with open(script_fname, 'a') as f:
                f.write('@ECHO OFF\n')
                f.write('call '+anaconda_activate_dir+' '+env_dir+'\n')
                f.write('call cd '+EQL_dir+'\n')
                f.write('call '+EQL_drive+'\n')
                f.write('PAUSE\n')
                f.write(cmd+'\n')
                f.write('PAUSE')
            id += 1
        
    script_fname_all = os.path.join(job_dir, "submit_all.bat")
    with open(script_fname_all, 'a') as f_all:
        f_all.write('@ECHO OFF\n')
        f_all.write('call '+anaconda_activate_dir+' '+env_dir+'\n')
        f_all.write('call cd '+EQL_dir+'\n')
        f_all.write('call '+EQL_drive+'\n')
        f_all.write('PAUSE\n')
        for i in range(len(cmd_all)):
            f_all.write(cmd_all[i]+'\n')
        f_all.write('PAUSE')
            
    print('Jobs succesfully generated.')
    

if __name__ == '__main__':

    train_val_file = 'example_data/F4data_train_val'
    test_file = 'example_data/F4data_test'
    input_fun = r'_model_selection_reduced\F4'
    
    anaconda_activate_dir = r'C:\ProgramData\Anaconda3\Scripts\activate.bat'
    env_dir = r'D:\GMi\condaenv\GMi_EQL'
    EQL_dir = r'D:\GMi\Arbeitsordner_GMi\SymbolicRegression\JupyterNotebook\EQL_Tensorflow-master'
    EQL_drive = r'D:'

    generate_jobs_bat(train_val_file, test_file, anaconda_activate_dir, env_dir, EQL_dir, EQL_drive, input_fun)
    

    
    
