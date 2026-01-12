# %%
import numpy as np
import dtw
import sofar as sf
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline as ip


# %% functions
def system_mismatch(ir_ref, ir_query):
    return np.linalg.norm(ir_ref - ir_query) / np.linalg.norm(ir_ref)


# %% Load and inspect HRIR
hrir = sf.read_sofa("HRIRs/scut/SCUT_KEMAR_radius_0.25.sofa")
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

# %% Calculate DTW between two HRIRs
ir_idx1 = np.where((hrir_angles == 270))[0][0]
ir_idx2 = ir_idx1 + 4

query = np.squeeze(hrir_data_horizon[ir_idx1, 0, 0:150])
reference = np.squeeze(hrir_data_horizon[ir_idx2, 0, 0:150])

target = np.squeeze(hrir_data_horizon[ir_idx1 + 2, 0, 0:150])

# stepPattern = dtw.rabinerJuangStepPattern(6, "c")
stepPattern = dtw.symmetricP2

alignment = dtw.dtw(query, reference, step_pattern=stepPattern, keep_internals=True)
alignment.plot(type="threeway")

dtw.dtwPlotTwoWay(alignment, xts=query, yts=reference, offset=-500)

print(stepPattern)
stepPattern.plot()

# %% Test warping
wq = dtw.warp(alignment, index_reference=False)
wt = dtw.warp(alignment, index_reference=True)


plt.figure()
plt.plot(reference, label="Reference", color="tab:blue")
plt.plot(query, label="Query", color="tab:orange")
plt.plot(query[wq], label="Warped Query", color="tab:blue", linestyle="--")
plt.title("Warping query")
plt.legend()

# plt.figure()
# plt.plot(reference, label="Reference", color="tab:blue")
# plt.plot(query, label="Query", color="tab:orange")
# plt.plot(reference[wt], label="Warped Reference", color="tab:orange", linestyle="--")
# plt.title("Warping reference")
# plt.legend()

# %% Inteprolation
# Linear interpolation of coeffs
h_interp_warped = (reference + query[wq]) / 2
h_interp_direct = (query + reference) / 2

# Find updated indoces for de-warping
displacement = np.arange(len(query)) - wq
idx_dewarping = np.arange(len(query)) - displacement * 0.5

# %% Apply spline interpolation to get samples at integer-indices
spline_wapred = ip(idx_dewarping, h_interp_warped)
h_dtw_dewarped = spline_wapred(np.arange(len(h_interp_warped)))

# %% Plot reference vs inteprolated
D_direct = system_mismatch(target, h_interp_direct)
D_dtw = system_mismatch(target, h_dtw_dewarped)
D_ref = system_mismatch(target, target)

plt.figure()
plt.plot(target, label=f"Reference (D={D_ref:.2f})", color="tab:blue")
plt.plot(h_dtw_dewarped, label=f"DTW Interpolated (D={D_dtw:.2f})", color="tab:orange")
plt.plot(
    h_interp_direct, label=f"Direct Interpolated (D={D_direct:.2f})", color="tab:green"
)
plt.legend()
plt.grid()

# %%
plt.show()
