import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--files', nargs='*')
args = parser.parse_args()
list_files = args.files


# read csv file
df = pd.concat([pd.read_csv(i, sep='  ', names=['case','index', 'call item', 'time (sec)'], engine='python') for i in list_files], ignore_index=True)
df['time (sec)'] = df['time (sec)'].astype(float)
df['index']="data[" + df['index'].str.replace(":", " : ").str.replace(",",", ").str.replace(",  :",", :") + ["]"]
df["method"] = df["call item"].apply(lambda s: " ".join(s.split()[0:2]))

# set up the plot
sns.set_theme(style="ticks")
sns.set(style="whitegrid", context="notebook", rc={"xtick.bottom" : True, 'xtick.minor.visible': True})
sns_palette = "colorblind"
sns_colors = sns.color_palette(sns_palette, n_colors=10)
sns_colors[0]=sns_colors[1]
sns_colors[1]=sns_colors[9]
sns.set_palette(sns_colors)
f = plt.figure(figsize=(22, 10))
wspace = 0.05
gs0 = gridspec.GridSpec(1, 1, figure=f, width_ratios=[7], wspace=wspace)
ax1 = f.add_subplot(gs0[0])

# setup log scale
log_axis = True
if log_axis:
    line_at = 0.0316
    df.loc[df["time (sec)"] <= 0.002,["time (sec)"]]=line_at
    ax1.set_xscale("log")
else:
    line_at = 0

# ylabel order sorted by ros3 read time
df["label"] = df["call item"].apply(lambda s: " ".join(s.split()[2:]).title() )
y_label_order_inst = sorted([i for i in df["label"].unique() if i != ""])
df["label"] = df.apply(lambda s: s["label"] or s["case"]+"\n"+s["index"], axis=1)
y_label_order = df[df['call item']=='time h5py_ros3'].groupby('label', as_index=False).mean().sort_values("time (sec)")["label"].to_list()[::-1]
y_label_order += y_label_order_inst

# plot times
ax1.axvline(line_at, color="k", clip_on=False, zorder=0)
hue_order = sorted(df["method"].unique())
sns.boxplot(x='time (sec)', y='label', data=df, hue='method', hue_order=hue_order, width=0.52, linewidth=1, fliersize=0, ax=ax1, order=y_label_order)
legend_lines_ax1, legend_labels_ax1 = ax1.get_legend_handles_labels()  # save legend
sns.swarmplot(x='time (sec)', y='label', data=df, size=7.5, hue='method', hue_order=hue_order, ax=ax1, dodge=False, orient='h', order=y_label_order)
# remove swarmplot legend
legend_ax1 = ax1.legend()
legend_ax1.remove()
# restore legend
legend_ax1 = ax1.legend(legend_lines_ax1, legend_labels_ax1, loc="lower right")
# add labels
xticks = np.array([line_at, *ax1.get_xticks()])
xticks.sort()
ax1.set_xticks(xticks)
xticks_labels = [i if i!=line_at else "<0.002" for i in xticks]
ax1.set_xticklabels(xticks_labels)
ax1.set_xlabel("Time (Sec)\n", x=0.5)
ax1.set_ylabel("Dataset: /processing/ecephys/SpikeWaveforms8/data\n Shape: [308092, 32, 1],     Chunk size: [9628, 2, 1],     Chunk count ~ [32, 16, 1]", labelpad=25)
legend_ax1.set_title("")
# set xbound
xbound = ( -0.18, df["time (sec)"].max()*1.25)
if log_axis:
    ax1.set_xbound([line_at/(10**(np.log10(xbound[1]/line_at)/(xbound[1]/0.18))),xbound[1]])
else:
    ax1.set_xbound(xbound)
# Tweak the visual presentation
ax1.xaxis.grid(True, which='both')
ax1.yaxis.grid(True)
sns.despine(left=True, bottom=False)

f.savefig('fig.png', bbox_inches='tight', dpi=200)
