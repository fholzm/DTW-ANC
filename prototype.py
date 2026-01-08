# %%
import numpy as np
import pyfar as pf
import sofar as sf
import matplotlib.pyplot as plt

# %% Load and inspect HRIR
hrir = sf.read_sofa("HRIRs/scut/SCUT_KEMAR_radius_1.sofa")
hrir.inspect()

# %%
hrir.list_dimensions

# %% Find HRIRs with sources on horizontal plane
source_pos = hrir.SourcePosition
horizontal_indices = np.where(source_pos[:, 1] == 0)[0]

hrir_data_horizon = hrir.Data_IR[horizontal_indices]
hrir_angles = hrir.SourcePosition[horizontal_indices, 0]

hrir_data_horizon = hrir_data_horizon[np.argsort(hrir_angles)]
hrir_angles = np.sort(hrir_angles)
fs = hrir.Data_SamplingRate

# %% Plot extracted HRIRs
t_lim_plot = 128
lim = -40

t_axis = np.arange(hrir_data_horizon.shape[2]) / fs * 1000
X, Y = np.meshgrid(t_axis[:t_lim_plot], hrir_angles)

data_log = np.abs(np.squeeze(hrir_data_horizon[:, 0, :t_lim_plot]))

data_log = 20 * np.log10(np.abs(np.squeeze(hrir_data_horizon[:, 0, :t_lim_plot])))
data_log -= np.max(data_log)
data_log[data_log < lim] = lim
data_log = data_log

plt.figure()
plt.pcolormesh(X, Y, data_log, shading="auto", cmap="gray")
plt.xlabel("time / ms")
plt.ylabel("angle / °")


# %%
contralat_angle_idx = np.where((hrir_angles >= 200) & (hrir_angles <= 280))[0]

plotting_offset = 1500
data_plot = np.squeeze(hrir_data_horizon[contralat_angle_idx, 0])
# data_plot += plotting_offset * np.arange(data_plot.shape[0])[:, np.newaxis]

x = t_axis[50:128]
y = hrir_angles[contralat_angle_idx]
Z = data_plot[:, 50:128]

ax = plt.figure(figsize=(6, 12)).add_subplot(projection="3d")
ax.view_init(elev=35, azim=90, roll=0)

# ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
for i in range(Z.shape[0]):
    ax.plot(x, y[i], Z[i, :], color="black")
plt.grid()

# idx = 0
# for ii in contralat_angle_idx:
#     idx += 1
#     plt.subplot(len(contralat_angle_idx) + 1, 1, idx)
#     plt.plot(t_axis[50:128], np.squeeze(data_plot[idx - 1, 50:128].T))
#     plt.text(2.5, 0.5 * np.max(data_plot[idx - 1, 50:128]), f"{hrir_angles[ii]}°")
#     plt.grid()

# %%
