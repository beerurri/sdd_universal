import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from check_results import parse_filename
from build_filename import build_filename

results = [parse_filename(file) for file in os.listdir(os.path.join(os.getcwd(), 'results'))]

results = [
    res for res in results
    if res['task'] == 'narma' and
    res['case'] == 'NARMA-TAU' and
    res['constant_theta'] == True and
    res['every_second'] == False]

results = sorted(results, key=lambda x: x['tau0'])
taus = [d['tau0'] for d in results]

results = [build_filename(res) for res in results]

# _ = [print(res) for res in results]
# print(f'{len(results)} vs {len(SF_tau_0)}')

mean = list()
std = list()

for file in results:
    source = np.loadtxt(os.path.join(os.getcwd(), 'results', file), delimiter=',')
    mean.append(source.mean())
    std.append(source.std())

mean = np.array(mean)
std = np.array(std)

# fig = plt.figure(figsize=(6, 4), dpi=200)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('NARMA-TAU')
ax2 = ax1.twiny()

ax1.plot(taus, mean, '-')
ax1.fill_between(taus, mean - std, mean + std, alpha=0.3)
ax1.set_ylabel('$NRMSE$')

ax1.set_xlabel('$\\tau_0$')
ax1.grid()

def tick_func(x):
    return [int(10 * _x) for _x in x]

new_tick_positions = [x for x in np.linspace(min(taus), max(taus), int(max(taus)), endpoint=True) if x % 5 == 0]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_positions)
ax2.set_xticklabels(tick_func(new_tick_positions))
ax2.set_xlabel('$N$')
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'py-scripts', 'plots', 'results', 'NARMA-TAU.png'))

# fig = plt.figure(figsize=(6, 4), dpi=200)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title('NARMA-TAU')
ax2 = ax1.twiny()
ax1.plot(taus, np.log10(mean), '-')
ax1.fill_between(taus, np.log10(mean - std), np.log10(mean + std), alpha=0.3, label='StDev')
ax1.set_ylabel('$\\log_{10}(NRMSE)$')
ax1.set_xlabel('$\\tau_0$')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(new_tick_positions)
ax2.set_xticklabels(tick_func(new_tick_positions))
ax2.set_xlabel('$N$')
ax1.grid()
fig.tight_layout()
fig.savefig(os.path.join(os.getcwd(), 'py-scripts', 'plots', 'results', 'NARMA-TAU-log10.png'))

plt.show()