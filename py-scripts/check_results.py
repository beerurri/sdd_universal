import os
from run_scheduler import make_commands


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
    _constant_theta = 'true' in filename.split('_constant-theta_')[1]

    return {
        'task': _task,
        'case': _case,
        'theta': _theta,
        'beta': _beta,
        'tau0': _tau0,
        'every_second': _every_second,
        'constant_theta': _constant_theta
    }


def parse_cmd(cmd: list):
    '''
    cmd = [
        EXE_PATH,
        "--task", task,
        "--case", case,
        "--theta", str(theta),
        "--beta", str(beta),
        "--tau-0", str(tau_0)
    ]
    if constant_theta:
        cmd.append("--constant-theta")
    if every_second:
        cmd.append("--every-second")
    return cmd
    '''

    _task = cmd[2]
    _case = cmd[4]
    _theta = float(cmd[6])
    _beta = float(cmd[8])
    _tau0 = float(cmd[10])
    _constant_theta = '--constant-theta' in cmd
    _every_second = '--every-second' in cmd

    return {
        'task': _task,
        'case': _case,
        'theta': _theta,
        'beta': _beta,
        'tau0': _tau0,
        'every_second': _every_second,
        'constant_theta': _constant_theta
    }


def remove_mc_duplicates(results):
    seen = set()
    unique = []
    for d in results:
        key = tuple(sorted(dict(d).items()))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


results = remove_mc_duplicates([parse_filename(file) for file in os.listdir(os.path.join(os.path.join(os.getcwd(), 'results')))])
commands = [parse_cmd(cmd) for cmd in make_commands()]
missing = list()
cases = ['SF-TAU', 'NARMA-TAU', 'MC-TAU', 'SF-THETA', 'NARMA-THETA', 'MC-THETA', 'SF-2D', 'NARMA-2D', 'MC-2D']

for case in cases:
    tmp_cmd = [cmd for cmd in commands if cmd['case'] == case]
    tmp_res = [res for res in results if res['case'] == case]

    if case in ['SF-TAU', 'NARMA-TAU', 'MC_TAU']:
        tau0_res = {d['tau0'] for d in tmp_res}
        missing.extend([d for d in tmp_cmd if d['tau0'] not in tau0_res])

    elif case in ['SF-THETA', 'NARMA-THETA', 'MC-THETA']:
        theta_res = {d['theta'] for d in tmp_res}
        missing.extend([d for d in tmp_cmd if d['theta'] not in theta_res])
        
    elif case in ['SF-2D', 'NARMA-2D', 'MC-2D']:
        theta_beta_res = {(d['theta'], d['beta']) for d in tmp_res}
        missing.extend([d for d in tmp_cmd if (d['theta'], d['beta']) not in theta_beta_res])


_ = [print(ms) for ms in missing]
print(f'missing len: {len(missing)}')
            
    


# _ = [print(res) for res in results]

# _ = [print(parse_cmd(cmd)) for cmd in commands]

# _ = [print(parse_filename(file)) for file in results_files]

# parsed = [parse_filename(file) for file in results_files]

# print(sorted([file['tau0'] for file in parsed if file['task'] == 'mc']))