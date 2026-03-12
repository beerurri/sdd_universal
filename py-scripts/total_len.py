from params import MC_tau_0, SF_tau_0, NARMA_tau_0, MC_thetas, SF_thetas, NARMA_thetas, theta_beta

total = 4 * len(MC_tau_0) + len(SF_tau_0) + len(NARMA_tau_0) + 4 * len(MC_thetas) + len(SF_thetas) + len(NARMA_thetas) + 6 * len(theta_beta)

print(total)