# Scenario I: different beta, everything else stay the same
ARGS1 = dict(
        time_step=50,
        number_of_cluster=3,
        S0_list=[0.97, 0.97, 0.97],
        I0_list=[0.01, 0.01, 0.01],
        beta_list=[0.1, 0.15, 0.2],
        sigma_list=[0.2, 0.2, 0.2],
        gamma_list=[0.2, 0.2, 0.2],
        tau_matrix=[[0.15, 0.15, 0.15], [0.15, 0.15, 0.15], [0.15, 0.15, 0.15]],
        phi_matrix=[[0.15, 0.15, 0.15], [0.15, 0.15, 0.15], [0.15, 0.15, 0.15]],
        objective_weight=[1, 1, 1],
        parallel=False,
        lr=10e-7,
        method='L-BFGS-B'
)

# Scenario II: different weights, everything else stay the same
ARGS2 = dict(
        number_of_cluster=3,
        S0_list=[0.97, 0.97, 0.97],
        I0_list=[0.01, 0.01, 0.01],
        beta_list=[0.2, 0.2, 0.2],
        sigma_list=[0.2, 0.2, 0.2],
        gamma_list=[0.2, 0.2, 0.2],
        tau_matrix=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        phi_matrix=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        time_step=50,
        objective_weight=[1, 2, 3],
        parallel=False,
        lr=10e-7,
        method='L-BFGS-B'
)