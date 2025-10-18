# =====================================================
# WhatsApp Traffic Model – Multi-Type ON–OFF VBR Source
# =====================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile

# ---------- 1. Load real Wireshark data ----------
# Make sure your CSV (data_wp_filtered.csv) is in the same directory
try:
    data = pd.read_csv("data_wp_filtered.csv")
except FileNotFoundError:
    print("Error: data_wp_filtered.csv not found in current directory.")
    print("Please ensure the file exists and try again.")
    exit(1)

# Standardize column names and identify time/size
cols = [c.lower() for c in data.columns]
data.columns = cols
time_col = [c for c in cols if 'time' in c][0]
size_col = [c for c in cols if 'length' in c or 'size' in c][0]

real_time = data[time_col].values
real_size = data[size_col].values

# ---------- 2. Define ON–OFF VBR model parameters ----------
λ = 15              # packets per second during ON
T_on_mean = 5         # mean ON duration [s]
T_off_mean = 3        # mean OFF duration [s]
sim_duration = real_time.max()

# Define data-type probabilities and mean sizes
types = ["text", "audio", "file"]
p = [0.8, 0.1, 0.1]  # probability of each data type

# Average packet sizes (bytes)
mean_sizes = {
    "text": 70,
    "audio": 500,
    "file": 4000
}
# Lognormal parameters per type
sigma_w = 0.5
mu_w = {k: np.log(v) for k, v in mean_sizes.items()}

# ---------- 3. Generate synthetic ON–OFF trace ----------
t, times, sizes, types_used = 0, [], [], []
while t < sim_duration:
    # OFF period (no packets)
    t += np.random.exponential(T_off_mean)
    
    # ON period
    t_on = np.random.exponential(T_on_mean)
    while t_on > 0:
        a = np.random.exponential(1/λ)  # exponential inter-arrival
        msg_type = np.random.choice(types, p=p)
        w = np.random.lognormal(mu_w[msg_type], sigma_w)  # workload by type
        t += a
        t_on -= a
        if t < sim_duration:
            times.append(t)
            sizes.append(w)
            types_used.append(msg_type)

synthetic = pd.DataFrame({'time': times, 'size': sizes, 'type': types_used})

# ---------- 4. Create output folder ----------
output_dir = "whatsapp_plots"
os.makedirs(output_dir, exist_ok=True)

# ---------- 5. Generate plots ----------

# --- Scatterplot Comparison ---
plt.figure(figsize=(10,5))
plt.scatter(real_time, real_size, s=10, alpha=0.5, label="Real WhatsApp trace")
plt.scatter(synthetic['time'], synthetic['size'], s=10, alpha=0.5, label="Synthetic data")
plt.xlabel("Time [s]")
plt.ylabel("Packet size [bytes]")
plt.title("WhatsApp Traffic: Real vs. Synthetic Trace (Scatterplot)")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/scatterplotComparison.png", dpi=300)
plt.close()


# --- Scatterplot Synthetic distribution ---
plt.figure(figsize=(10,5))
for msg_type, group in synthetic.groupby('type'):
    plt.scatter(group['time'], group['size'], s=10, alpha=0.5, label=f"Synthetic ({msg_type})")
plt.xlabel("Time [s]")
plt.ylabel("Packet size [bytes]")
plt.title("Synthetic Data Distribution")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/scatterplotSyntheticDistribution.png", dpi=300)
plt.close()

# --- CDF of Packet Sizes ---
def cdf(data):
    return np.sort(data), np.arange(1, len(data)+1) / len(data)
r_sorted, r_cdf = cdf(real_size)
s_sorted, s_cdf = cdf(synthetic['size'])
plt.figure(figsize=(7,5))
plt.plot(r_sorted, r_cdf, label="Real data")
plt.plot(s_sorted, s_cdf, label="Synthetic model")
plt.xlabel("Packet size [bytes]")
plt.ylabel("CDF")
plt.title("CDF of Packet Sizes: Real vs. Synthetic")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/cdf.png", dpi=300)
plt.close()

# --- QQ-Plot ---
real_sorted = np.sort(real_size)
synthetic_sorted = np.sort(synthetic['size'])
min_len = min(len(real_sorted), len(synthetic_sorted))
plt.figure(figsize=(6,6))
plt.scatter(real_sorted[:min_len], synthetic_sorted[:min_len], s=10, alpha=0.6)
plt.plot([0, max(real_sorted)], [0, max(real_sorted)], 'r--')
plt.xlabel("Real quantiles")
plt.ylabel("Synthetic quantiles")
plt.title("QQ-Plot: Real vs. Synthetic Packet Sizes")
plt.grid(True)
plt.savefig(f"{output_dir}/qqplot.png", dpi=300)
plt.close()

# --- Boxplot log10---
plt.figure(figsize=(6,5))
# Use log10 transformation to reduce skew and make distributions comparable
real_box = np.log10(real_size)
synthetic_box = np.log10(synthetic['size'])
plt.boxplot([real_box, synthetic_box], labels=['Real', 'Synthetic'])
plt.ylabel("Packet size [bytes]")
plt.title("Boxplot Comparison of Packet Sizes (log10 scale)")
plt.grid(True)
plt.savefig(f"{output_dir}/boxplot.png", dpi=300)
plt.close()

# --- Histogram / PDF Overlay ---
plt.figure(figsize=(7,5))
plt.hist(real_size, bins=50, density=True, alpha=0.6, label='Real')
plt.hist(synthetic['size'], bins=50, density=True, alpha=0.6, label='Synthetic')
plt.xlabel("Packet size [bytes]")
plt.ylabel("Probability density")
plt.title("Histogram Comparison of Packet Sizes")
plt.legend()
plt.savefig(f"{output_dir}/histogram.png", dpi=300)
plt.close()

# --- Inter-arrival Time Distribution ---
real_inter = np.diff(real_time)
synthetic_inter = np.diff(synthetic['time'])
plt.figure(figsize=(7,5))
plt.hist(real_inter, bins=50, density=True, alpha=0.6, label='Real')
plt.hist(synthetic_inter, bins=50, density=True, alpha=0.6, label='Synthetic')
plt.xlabel("Inter-arrival time [s]")
plt.ylabel("Probability density")
plt.title("Inter-arrival Time Distribution Comparison")
plt.legend()
plt.savefig(f"{output_dir}/interarrival.png", dpi=300)
plt.close()

# --- Throughput over Time ---
bin_width = 1
real_rate = pd.Series(real_time).groupby((real_time // bin_width)).size() / bin_width
synthetic_rate = pd.Series(synthetic['time']).groupby((synthetic['time'] // bin_width)).size() / bin_width
plt.figure(figsize=(10,4))
plt.plot(real_rate.index, real_rate.values, label='Real')
plt.plot(synthetic_rate.index, synthetic_rate.values, label='Synthetic')
plt.xlabel("Time [s]")
plt.ylabel("Packets/s")
plt.title("Instantaneous Packet Rate (Throughput over Time)")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/throughput.png", dpi=300)
plt.close()

# ---------- 6. Zip all plots ----------
zip_path = "whatsapp_plots.zip"
# Remove existing zip if it exists
if os.path.exists(zip_path):
    os.remove(zip_path)

with ZipFile(zip_path, 'w') as zipf:
    for file in os.listdir(output_dir):
        zipf.write(os.path.join(output_dir, file), arcname=file)

print(f"ZIP file created: {zip_path}")
print("\nFiles included:")
for file in os.listdir(output_dir):
    print(f" - {file}")

# ---------- 7. Compute and print offered traffic ----------
# Duty cycle (d) = TON / (TON + TOFF)
d = T_on_mean / (T_on_mean + T_off_mean)
# Average workload (weighted mean)
E_W = sum([p[i] * mean_sizes[t] for i, t in enumerate(types)])
# This line multiplies each type's probability (p[i]) with its mean size (mean_sizes[t])
# and sums them up, which is exactly the weighted average we want

# Offered load
Y_bar = λ * E_W * d
print(f"\nAverage offered traffic (λ * E[W] * d): {Y_bar:.2f} bytes/s ≈ {Y_bar*8/1000:.2f} kbit/s")