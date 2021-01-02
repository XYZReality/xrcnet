'''
this script is to plot our result with sparse ncnet in Hpatches dataset
'''
import numpy as np
import matplotlib.pyplot as plt

# create a dictionary that contain all result
files = {
    'XRCNet 3k' : 'cache-top/xrcnet_3000_2000.txt',
    'XRCNet 1.6k' : 'cache-top/xrcnet_1600_2000.txt',
    'DualRC-Net' : 'cache-top/dualrc-net_1600_2000.txt',
    'Sparse-NCNet (1600 H+S)' : 'cache-top/ncnet.sparsencnet_1600_hard_soft_2k.npy',
    'NCNet (3200 H)' : 'cache-top/ncnet.densencnet_1600_hard_2k.npy',
    'D2-Net' : 'cache-top/d2-net.npy',
    'DELF' : 'cache-top/delf.npy',
    'R2D2' : 'cache-top/r2d2.npy',
    'SP': 'cache-top/superpoint.npy',
    'SP+SG': 'cache-top/sp_sg_ImgSize1600_MaxNum2000_SGThres0_NMS2.txt',
}

markertypes = ['o', '*', '^', '>', 'h', 'p', '+', 'D', '|', 'x', '.', '<', 'X', 'd', '2', 'P', '8', 'x']
markertypes = markertypes[::-1]

markercolors = [
    '#4363d8',
    '#f58231',
    '#911eb4',
    '#42d4f4',
    '#f032e6',
    '#bfef45',
    '#fabed4',
    '#FF0000',
    '#ffe119',
    '#3cb44b',
    '#e6194B',
    "#BC5090",
    "#A66249",
    "#3CAEA3",
    "#ED553B",
    "#F6D55C",
    "#20639B",
    "#173F5F"
]
markercolors = markercolors[::-1]

top_k = 6

def read_npy(file):
    data = np.load(file, allow_pickle=True)
    type, num = np.unique(data[2][0], return_counts=True)
    count_dict = dict(zip(type, num))
    num_i = count_dict['i']
    num_v = count_dict['v']

    MMA_i_ = np.array(list(data[0].values())[:10])
    MMA_v_ = np.array(list(data[1].values())[:10])
    MMA_i = MMA_i_ / num_i
    MMA_v = MMA_v_ / num_v
    MMA = (MMA_i_ + MMA_v_) / (num_i + num_v)

    MMA_i = MMA_i[:top_k]
    MMA_v = MMA_v[:top_k]
    MMA = MMA[:top_k]

    return MMA_i, MMA_v, MMA, np.round(np.trapz(MMA_i), 4), np.round(np.trapz(MMA_v), 4), np.round(np.trapz(MMA), 4)

def read_txt(our_file):
    with open(our_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if line[0] == 'MMA_i':
                MMA_i = np.array(line[1:], dtype=float)
            if line[0] == 'MMA_v':
                MMA_v = np.array(line[1:], dtype=float)
            if line[0] == 'MMA':
                MMA = np.array(line[1:], dtype=float)
    
    MMA_i = MMA_i[:top_k]
    MMA_v = MMA_v[:top_k]
    MMA = MMA[:top_k]

    return MMA_i, MMA_v, MMA, np.round(np.trapz(MMA_i), 4), np.round(np.trapz(MMA_v), 4), np.round(np.trapz(MMA), 4)

x = np.linspace(1, top_k, num=top_k ,endpoint=True)

legend_fontsize = 16
linewidth = 4
markersize = 8
title_size = 25
label_size = 18
tick_size = 20
position = 0.15

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(27, 7.5))

for i, (key, path) in enumerate(files.items()):
    if path[-3:] == 'txt':
        MMA_i, MMA_v, MMA, MMA_i_sum, MMA_v_sum, MMA_sum = read_txt(path)
    elif path[-3:] == 'npy':
        MMA_i, MMA_v, MMA, MMA_i_sum, MMA_v_sum, MMA_sum = read_npy(path)
    else:
        assert 'File format error'

    MMA_i_label = key
    MMA_v_label = key
    MMA_label = key
    axes[0].plot(x, MMA_i, marker=markertypes[i], color=markercolors[i], label=MMA_i_label, linewidth=linewidth, markersize=markersize)
    axes[1].plot(x, MMA_v, marker=markertypes[i], color=markercolors[i], label=MMA_v_label, linewidth=linewidth, markersize=markersize)
    axes[2].plot(x, MMA, marker=markertypes[i], color=markercolors[i], label=MMA_label, linewidth=linewidth, markersize=markersize)

axes[0].set_title('Illumination', size=title_size)
axes[0].set_xlabel('threshold [px]', fontsize=label_size)
axes[0].set_ylabel('MMA', fontsize=label_size)
axes[0].set_xlim([1, top_k])
axes[0].set_ylim([0, 1])
axes[0].grid()
axes[0].tick_params(axis='x', labelsize=tick_size)
axes[0].tick_params(axis='y', labelsize=tick_size)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[::-1], labels[::-1], title='Line', loc='best', fontsize=legend_fontsize)

axes[1].set_title('Viewpoint', size=title_size)
axes[1].set_xlabel('threshold [px]', fontsize=label_size)
axes[1].set_ylabel('MMA', fontsize=label_size)
axes[1].set_xlim([1, top_k])
axes[1].set_ylim([0, 1])
axes[1].grid()
axes[1].tick_params(axis='x', labelsize=tick_size)
axes[1].tick_params(axis='y', labelsize=tick_size)

axes[2].set_title('Overall', size=title_size)
axes[2].set_xlabel('threshold [px]', fontsize=label_size)
axes[2].set_ylabel('MMA', fontsize=label_size)
axes[2].set_xlim([1, top_k])
axes[2].set_ylim([0, 1])
axes[2].grid()
axes[2].tick_params(axis='x', labelsize=tick_size)
axes[2].tick_params(axis='y', labelsize=tick_size)

plt.show()

