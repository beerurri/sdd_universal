import os


results_files = os.listdir(os.path.join(os.path.join(os.getcwd(), 'results')))

def parse_filename(filename: str):
    '''
    santa_case_SF-TAU_theta_0.12_beta_0.5_tau0_50_every_second_false_constant-theta_true.csv
    '''

    _task = filename.split('_case_')[0]
    _case = filename.split('_case_')[1].split('_theta_')[0]
    _theta = float(filename.split('_theta_')[1].split('_beta_')[0])
    _beta = float(filename.split('_beta_')[1].split('_tau0_')[0])
    _tau0 = float(filename.split('_tau0_')[1].split('_every_second_')[0])
    _every_second = filename.split('_every_second_')[1].split('_constant-theta_')[0] == 'true'
    _constant_theta = filename.split('_constant-theta_')[1] == 'true'

    return {
        'task': _task,
        'case': _case,
        'theta': _theta,
        'beta': _beta,
        'tau0': _tau0,
        'every_second': _every_second,
        'constant_theta': _constant_theta
    }


parsed = [parse_filename(file) for file in results_files]

print(sorted([file['tau0'] for file in parsed if file['task'] == 'mc']))