import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

nwbfile_local = 'data/ecephys-001.nwb'
hfile = h5py.File(nwbfile_local, mode='r')

# get dataset sizes
dset_storagesize = {}


def _info(name, hobj):
    if isinstance(hobj, h5py.Dataset):
        dset_storagesize[name] = hobj.id.get_storage_size()


hfile.visititems(_info)

# read csv file
logfile = 'logs/logfile-001.csv'
df = pd.read_csv(logfile)
_, _, _name1, _name2, _name3 = df.columns
_name4 = 'h5py Read Time'
df[_name4] = df[_name2]+df[_name3]
df['size'] = df['name'].map(dset_storagesize)
id_vars = ['Type', 'name', 'size']
df = df.melt(id_vars=id_vars, var_name='test', value_name='time')

# get target sizes for soft links
df_softlinks = df[df["size"].isnull()]
for name in df_softlinks['name'].unique():
    _info(name, hfile[name])

# update sizes
df['size'] = df['name'].map(dset_storagesize)

# select a subset of datasets by size
df = df[df['Type'] == 'Dataset']
df = df[df['size'] > 2**20*0.05]

# set up the plot and the grid
# |ax11 ax12|
# |ax21 ax22|
sns_palette_ = "deep"
sns.set(style="whitegrid", context="talk")
sns_colors_ = sns.color_palette(sns_palette_, n_colors=20)
sns_colors_[4] = sns_colors_[8]
sns.set_palette(sns_colors_)

f = plt.figure(figsize=(18, 25))
gs0 = gridspec.GridSpec(2, 1, figure=f, height_ratios=[7, 1])
wspace = 0.05
gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=wspace)
ax11 = f.add_subplot(gs00[0, 0])
ax12 = f.add_subplot(gs00[0, 1])
gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=wspace)
ax21 = f.add_subplot(gs01[0, 0])
ax22 = f.add_subplot(gs01[0, 1])

break_at = 1000  # time column
df_ax11 = df[df["time"] < break_at]
df_ax12 = df[df["time"] >= break_at]
ax11.axvline(0, color="k", clip_on=False, zorder=0)

# plot times
hue_order = [_name1, _name2, _name3, _name4]
yticks_name_order = df[df["test"] == _name1].groupby(["time", "name"], as_index=False).mean()["name"].to_list()[::-1]
ax11 = sns.pointplot(x='time', y='name', hue='test', ax=ax11, data=df_ax11, order=yticks_name_order,
                     dodge=True, hue_order=hue_order)
# keep legend info
legend_lines_ax11, legend_labels_ax11 = ax11.get_legend_handles_labels()
legend_ax11 = ax11.legend()
legend_ax11.remove()

# plot times
sns.pointplot(x='time', y='name', hue='test', ax=ax12, data=df_ax12, order=yticks_name_order,
              dodge=True, hue_order=hue_order)
# keep bounds
xbound_ = ax12.get_xbound()
ybound_ = ax12.get_ybound()
# remove ax12 legend
legend_ax12 = ax12.legend()
legend_ax12.remove()

# add break lines
d = .005
kwargs = dict(transform=ax11.transAxes, color='k', clip_on=False)
ax11.plot((1-d, 1+d), (-d, +d), scalex=False, scaley=False, **kwargs)
# add hline on xaxis
ax11.plot((0, 1), (0, 0), scalex=False, scaley=False, **kwargs, zorder=max([_.zorder for _ in ax11.get_children()]))

kwargs.update(transform=ax12.transAxes)
ax12.plot((-d, +d), (-d, +d), scalex=False, scaley=False, **kwargs)
# add hline on xaxis
ax12.plot((0, 1), (0, 0), scalex=False, scaley=False, **kwargs, zorder=max([_.zorder for _ in ax12.get_children()]))

# fix bounds
ax12.set_ybound(ybound_)
ax12.set_xbound(xbound_)

# bring Zarr Read Time to the front
ax11.collections[0].zorder = ax11.collections[-1].zorder+1
ax12.collections[0].zorder = ax11.collections[-1].zorder+1

# add legend
legend_ax12 = ax12.legend(legend_lines_ax11, legend_labels_ax11, loc="lower right")
legend_ax12.set_title("")

# add boxplot
ax21 = sns.boxplot(x='time', y='test', ax=ax21, data=df)
ax21.set_xbound(ax11.get_xbound())
ax22 = sns.boxplot(x='time', y='test', ax=ax22, data=df)
ax22.set_xbound(ax12.get_xbound())

# add labels
ax11.set_yticklabels(yticks_name_order)
ax11.set_ylabel("Neuropixel Datasets")
ax11.set_xlabel("Time (Sec)", x=1+wspace/2)
ax21.set_xlabel("Time (Sec)", x=1+wspace/2)
ax21.set_ylabel("")

for ax in [ax12, ax22]:
    ax.set_yticklabels("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

for ax in [ax11, ax21]:
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=False)
ax11.figure.tight_layout()

ax11.figure.savefig('Zarr_Read_Time_Comparison2.png', bbox_inches='tight')
