import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from check_results import parse_filename
from build_filename import build_filename
from check_results import remove_mc_duplicates

fig, axs = plt.subplots(2, 3, constrained_layout=True)
fig.delaxes(axs[1, 2])

results = [parse_filename(file) for file in os.listdir(os.path.join(os.getcwd(), 'results'))]

# MC

mc_results = [
    res for res in results
    if res['task'] == 'mc' and
    res['case'] == 'MC-2D' and
    res['constant_theta'] == False and
    res['every_second'] == False
]

mc_results = remove_mc_duplicates(mc_results)
mc_results = sorted(mc_results, key=lambda x: (x['theta'], x['beta']))

betas = sorted({p['beta'] for p in mc_results})
thetas = sorted({p['theta'] for p in mc_results})

n_beta = len(betas)
n_theta = len(thetas)


def process_mc_file(file):
    with open(os.path.join(os.getcwd(), 'results', file), 'r') as f:
        source = f.read().strip('\n').split('\n')

    tmp = np.array([np.fromstring(string, sep=',').sum() for string in source if string])

    return tmp.mean() if len(tmp) else 0


mcs = dict()

for mc_type in ['LC', 'QC', 'CC', 'XC']:
    Z = np.zeros((n_theta, n_beta))

    for mc_res in mc_results:
        file = build_filename(mc_res, mc_type=mc_type)
        mean = process_mc_file(file)

        i = thetas.index(mc_res['theta'])
        j = betas.index(mc_res['beta'])

        Z[i, j] = mean

    mcs[mc_type] = Z


mcs['TC'] = mcs['LC'] + mcs['QC'] + mcs['CC'] + mcs['XC']

extent = [
    betas[0], betas[-1],
    thetas[0], thetas[-1]
]

dx = (betas[-1] - betas[0]) / n_beta
dy = (thetas[-1] - thetas[0]) / n_theta

x_centers = betas[0] + dx * (np.arange(n_beta) + 0.5)
y_centers = thetas[0] + dy * (np.arange(n_theta) + 0.5)

for i, mc_type in enumerate(mcs.keys()):
    ax = axs[i // 3, i % 3]
    im = ax.imshow(
        mcs[mc_type],
        origin='lower',
        aspect='auto',
        extent=extent,
        vmin=0
    )
    ax.set_title(mc_type)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\theta$')
    ax.set_xticks(x_centers)
    ax.set_xticklabels([f'{b:.2f}' for b in betas])

    ax.set_yticks(y_centers)
    ax.set_yticklabels([f'{t:.3f}' for t in thetas])
    plt.colorbar(im, ax=ax)


# NARMA

fig = plt.figure()

narma_results = [
    res for res in results
    if res['task'] == 'narma' and
    res['case'] == 'NARMA-2D' and
    res['constant_theta'] == False and
    res['every_second'] == False
]

betas = sorted({p['beta'] for p in narma_results})
thetas = sorted({p['theta'] for p in narma_results})

n_beta = len(betas)
n_theta = len(thetas)

narma_results = sorted(narma_results, key=lambda x: (x['theta'], x['beta']))

Z = np.zeros((n_theta, n_beta))

for narma_res in narma_results:
    file = build_filename(narma_res)

    mean = np.loadtxt(os.path.join(os.getcwd(), 'results', file), delimiter=',').mean()

    i = thetas.index(narma_res['theta'])
    j = betas.index(narma_res['beta'])

    Z[i, j] = mean

im = plt.imshow(
    Z,
    origin='lower',
    aspect='auto',
    extent=extent,
    vmin=0
)

plt.title('NARMA NRMSE')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\theta$')
plt.xticks(ticks=x_centers, labels=[f'{b:.2f}' for b in betas])

plt.yticks(ticks=y_centers, labels=[f'{t:.3f}' for t in thetas])
plt.colorbar()


# SF

fig = plt.figure()

santa_results = [
    res for res in results
    if res['task'] == 'santa' and
    res['case'] == 'SF-2D' and
    res['constant_theta'] == False and
    res['every_second'] == False
]

betas = sorted({p['beta'] for p in santa_results})
thetas = sorted({p['theta'] for p in santa_results})

n_beta = len(betas)
n_theta = len(thetas)

santa_results = sorted(santa_results, key=lambda x: (x['theta'], x['beta']))

Z = np.zeros((n_theta, n_beta))

for santa_res in santa_results:
    file = build_filename(santa_res)

    mean = np.loadtxt(os.path.join(os.getcwd(), 'results', file), delimiter=',').mean()

    i = thetas.index(santa_res['theta'])
    j = betas.index(santa_res['beta'])

    Z[i, j] = mean

im = plt.imshow(
    Z,
    origin='lower',
    aspect='auto',
    extent=extent,
    vmin=0,
    vmax=(Z.mean() / 4)
)

print(f'SF Z mean: {Z.mean()}')

plt.title('Santa Fe NMSE')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\theta$')
plt.xticks(ticks=x_centers, labels=[f'{b:.2f}' for b in betas])

plt.yticks(ticks=y_centers, labels=[f'{t:.3f}' for t in thetas])
plt.colorbar()


plt.show()