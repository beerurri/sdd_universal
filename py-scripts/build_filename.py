

def build_filename(d: dict, mc_type: str = None):
    '''
    {
        'task': _task,
        'case': _case,
        'theta': _theta,
        'beta': _beta,
        'tau0': _tau0,
        'every_second': _every_second,
        'constant_theta': _constant_theta
    }

    santa_case_SF-TAU_theta_0.12_beta_0.5_tau0_50_every_second_false_constant-theta_true.csv
    '''

    if d['theta'] % 1 == 0:
        d['theta'] = int(d['theta'])
    if d['beta'] % 1 == 0:
        d['beta'] = int(d['beta'])
    if d['tau0'] % 1 == 0:
        d['tau0'] = int(d['tau0'])


    res = f'{d["task"]}_case_{d["case"]}_theta_{d["theta"]}_beta_{d["beta"]}_tau0_{d["tau0"]}'
    res += f'_every_second_{"true" if d["every_second"] else "false"}'
    res += f'_constant-theta_{"true" if d["constant_theta"] else "false"}'
    res += f'_{mc_type}' if mc_type else ''
    res += '.csv'

    return res