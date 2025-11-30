import numpy as np
import matplotlib.pyplot as plt

# 1. Define Lambda Range (logarithmic scale from 10^-6 to 10^-1)
lambdas = np.logspace(-6, -1, 200)

# 2. Apply the Formula from your image
# U(lambda) = (1 + 55.5556 * lambda) / (1 + 92.9278 * lambda)
uptime = (1 + 55.5556 * lambdas) / (1 + 92.9278 * lambdas)

# 3. Plotting
plt.figure(figsize=(10, 6), dpi=100)
plt.semilogx(lambdas, uptime, color='#3b82f6', linewidth=2.5)
plt.title(r'System Uptime vs $\lambda$', fontsize=16)
plt.xlabel(r'$\lambda$ ', fontsize=14)
plt.ylabel('Uptime', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.4)

# Set limits to match the desired range
plt.ylim(0.58, 1.02)
plt.xlim(1e-6, 1e-1)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------

#  Monte Carlo Simulation 

# Parameters definition 
T_l = 10.0      
T_nc1 = 60.0   # non-critical error fix time case 1
T_nc2 = 180.0   # non-critical error fix time case 2
T_c = 30.0     

p_latent   = 0.6
p_noncrit  = 0.3
p_critical = 0.1

p_overwrite   = 0.3   # latent to overwritten
p_latent_fail = 0.7   # latent to critical

# Generate an exponential random variable with given mean.

rng = np.random.default_rng(seed=1)  # fixed seed for reproducibility

def exp_time(mean):
    return rng.exponential(mean)

# Uptime simulation function

def simulate_uptime(lam, reboot_mode="exp", n_events=100000):
  
    # State definition:
    # 1 = clean state
    # 2 = latent error
    # 3 = non-critical error
    # 4 = reboot state

    state = 1
    t = 0.0
    up_time = 0.0
    next_internal = np.inf  # time of next event

    for _ in range(n_events):
        # next bit flip arrives after Exp(mean=1/lam)
        inter_arrival = exp_time(1.0 / lam)
        t_arrival = t + inter_arrival

        # next event is either a bit flip or a transition
        t_next = min(t_arrival, next_internal)
        dt = t_next - t

        # count uptime (all states except reboot are up)
        if state != 4:
            up_time += dt

        t = t_next

        if t_next == t_arrival:
            if state == 4:
                # ignore flips during reboot
                continue

            r = rng.random()
            if r < p_latent:
                state = 2
                next_internal = t + exp_time(T_l)
            elif r < p_latent + p_noncrit:
                state = 3
                if rng.random() < 0.6:
                    next_internal = t + exp_time(T_nc1)
                else:
                    next_internal = t + exp_time(T_nc2)
            else:
                state = 4
                if reboot_mode == "exp":
                    next_internal = t + exp_time(T_c)
                else:  # "const"
                    next_internal = t + T_c

        else:
            if state == 2:
                # latent evolves
                if rng.random() < p_overwrite:
                    # safe overwrite and back to clean
                    state = 1
                    next_internal = np.inf
                else:
                    # becomes critical, so reboot
                    state = 4
                    if reboot_mode == "exp":
                        next_internal = t + exp_time(T_c)
                    else:
                        next_internal = t + T_c

            elif state == 3:
                # non-critical fixed, so reboot
                state = 4
                if reboot_mode == "exp":
                    next_internal = t + exp_time(T_c)
                else:
                    next_internal = t + T_c

            elif state == 4:
                # reboot finished, so back to clean
                state = 1
                next_internal = np.inf

    return up_time / t

#  Analytical BDP uptime
def uptime_bdp(lam):
    """Analytical uptime from the 3-state BDP model."""
    return (1 + 55.5556 * lam) / (1 + 92.9278 * lam)


#  Define range of lambda
lam_values = np.logspace(-6, -1, 12)

uptime_sim_exp   = np.array([simulate_uptime(l, reboot_mode="exp")   for l in lam_values])
uptime_sim_const = np.array([simulate_uptime(l, reboot_mode="const") for l in lam_values])
uptime_bdp_vals  = uptime_bdp(lam_values)


#  Plot 1: BDP analytical vs Monte Carlo simulation 
plt.figure(figsize=(10, 6))
plt.semilogx(lam_values, uptime_bdp_vals, label="BDP analytical", linewidth=2.5)
plt.semilogx(lam_values, uptime_sim_exp, 'o--', label="Simulation (real process, exp reboot)", linewidth=2)

plt.xlabel(r"$\lambda$")
plt.ylabel("Uptime")
plt.title("BDP Analytical Uptime vs Monte Carlo Simulation")
plt.grid(True, which="major", linestyle="--", alpha=0.6)
plt.ylim(0.45, 1.02)
plt.legend()
plt.show()

#  Plot 2: Exponential vs Constant reboot 
plt.figure(figsize=(10, 6))
plt.fill_between(lam_values, uptime_sim_const - 0.003, uptime_sim_const + 0.003,
                 color='red', alpha=0.3, label="Const reboot")

plt.semilogx(lam_values, uptime_sim_exp,   'o--', color='green', linewidth=2.5, label="Exp reboot")

plt.xlabel(r"$\lambda$")
plt.ylabel("Uptime")
plt.title("Impact of Reboot-Time Distribution on Uptime")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()