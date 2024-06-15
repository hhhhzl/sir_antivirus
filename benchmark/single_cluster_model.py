

# Scenario I: different beta, everything else stay the same
ARGS1 = dict(
    time_steps=100,
    initial_infected_prop=0.02,
    initial_recovered_prop=0.01,
    beta=0.1,
    sigma=0.1,
    gamma=0.1,
    delta_t=1,
    lr=10e-7,
    method='L-BFGS-B'
)