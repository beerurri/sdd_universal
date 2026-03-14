import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from check_results import parse_filename
from build_filename import build_filename
from check_results import remove_mc_duplicates

results = [parse_filename(file) for file in os.listdir(os.path.join(os.getcwd(), 'results'))]

results = [
    res for res in results
    if res['task'] == 'mc' and
    res['case'] == 'MC-TAU' and
    res['constant_theta'] == True and
    res['every_second'] == False]

results = sorted(results, key=lambda x: x['tau0'])
results = remove_mc_duplicates(results)
taus = [d['tau0'] for d in results]

mc_files = list() # 2d list
for mc_type in ['LC', 'QC', 'CC', 'XC']:
    mc_files.append([build_filename(res, mc_type=mc_type) for res in results])

# print(f'{len(results)} vs {len(SF_tau_0)}')

mean = list()
std = list()

for mc_type_files in mc_files:
    curr_mc_mean = list()
    curr_mc_std = list()
    for file in mc_type_files:
        with open(os.path.join(os.getcwd(), 'results', file), 'r') as f:
            source = f.read().strip('\n').split('\n')
            
        tmp = np.array([np.fromstring(string, sep=',').sum() for string in source if string])
        curr_mc_mean.append(tmp.mean())
        curr_mc_std.append(tmp.std())
    
    mean.append(np.array(curr_mc_mean))
    std.append(np.array(curr_mc_std))

mean_sum = np.array(mean).sum(axis=0)
mean_std = np.array(std).std(axis=0)

# fig = plt.figure(figsize=(6, 4), dpi=200)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('MC-TAU')
ax2 = ax1.twiny()

for i, mc_type in enumerate(['LC', 'QC', 'CC', 'XC']):
    ax1.plot(taus, mean[i], '-', label=mc_type)
    ax1.fill_between(taus, mean[i] - std[i], mean[i] + std[i], alpha=0.3)

ax1.plot(taus, mean_sum, '-', label='sum')
ax1.fill_between(taus, mean_sum - mean_std, mean_sum + mean_std, alpha=0.3)

ax1.set_ylabel('$MC$')

ax1.set_xlabel('$\\tau_0$')
ax1.grid()

def tick_func(x):
    return [int(10 * _x) for _x in x]

new_tick_positions = [x for x in np.linspace(min(taus), max(taus), int(max(taus)), endpoint=True) if x % 5 == 0]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_positions)
ax2.set_xticklabels(tick_func(new_tick_positions))
ax2.set_xlabel('$N$')

ax1.legend()
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'py-scripts', 'plots', 'results', 'MC-TAU.png'))

plt.show()