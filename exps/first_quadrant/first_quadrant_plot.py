import matplotlib.pyplot as plt
import numpy as np
import json
import os

label_fontsize = 16
ticks_fintsize = 12
legend_fontsize = 13
linewidth = 2
markersize = 10
SET_SIZES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300]


def plot_res(model2res, name2color, name2marker, title='', scale=1, loc='lower left'):
    plt.figure(figsize=(5 * scale, 4 * scale))
    x = np.arange(len(SET_SIZES))
    for name, vals in model2res.items():
        vals_mat = np.stack(list(vals.values()))
        m = np.mean(vals_mat, 0) # m[-3:] = [0.9918, 0.9876, 0.9802]
        s = np.std(vals_mat, 0)

        linestyle = ':' if name == 'GBeST' else '-'
        plt.plot(x, m, linestyle, color=name2color[name], marker=name2marker[name], markersize=markersize, linewidth=linewidth, label=name)
        plt.fill_between(x, m - s/2, m + s/2, color=name2color[name], alpha=0.2, interpolate=False)

    #plt.ylim([0.49,0.87])
    plt.ylabel('Accuracy', fontsize=label_fontsize)
    plt.xticks(x, SET_SIZES, fontsize=ticks_fintsize, rotation=45)
    plt.yticks(fontsize=ticks_fintsize)

    plt.xlabel('Test sets sizes', fontsize=label_fontsize)
    # plt.legend(fontsize=legend_fontsize, loc=loc) # loc='lower left'
    plt.tight_layout()
    plt.title(title)
    plt.show()


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


model2res = {'DeepSets_sum': None,
            'DeepSets_mean': None,
            'DeepSets_max': None,
            'GBeST': '/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2dim_multi/gbdt_2dim_multi_results_dump.json'}


###### 2dims ######
gbdt = read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2dim_multi/gbdt_2dim_multi_results_dump.json')
reg_gbdt = {k: v['reg'] for k, v in gbdt.items()}
set_gbdt = {k: v['set'] for k, v in gbdt.items()}
name2color = {'GBT': '#1f77b4',
              'GBeST': '#ff7f0e',
              'LSTM': '#2ca02c',
              'DeepSets (mean)': '#d62728',
              'DeepSets (sum)': '#9467bd',
              'DeepSets (max)':'#8c564b'}
'''
 '#e377c2',
 '#7f7f7f',
 '#bcbd22',
 '#17becf'
'''
name2marker = {'GBT': 'o',
              'GBeST': '^',
              'LSTM': 'D',
              'DeepSets (mean)': '*',
              'DeepSets (sum)': 'd',
              'DeepSets (max)':'s'}

model2res = {'GBT': reg_gbdt,
             'LSTM': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/rnn_set=20_train_varying_test_results_dump.json'),
             #'pi-SGD': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/rnn_jannossy_2dim_multi_results_dump.json'),
             'DeepSets (mean)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/deepsets_sum_2dim_multi_results_dump.json'),
             'DeepSets (sum)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/deepsets_mean_2dim_results_dump_old.json'),
             'DeepSets (max)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/deepsets_max_2dim_multi_results_dump.json'),
             'GBeST': set_gbdt}

# 'DDS_max': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/DDS/dds_max_2dim_multi_results_dump.json'),
             # 'DDS_sum': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/DDS/dds_sum_2dim_multi_results_dump.json'),
             # 'SIRE': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/SIRE/rnn_sire_1dim_multi_results_dump.json')}
# plot_res(model2res, name2color, name2marker, title='Q1 2dims', scale=1, loc='lower left')

###### 100dims ######
gbdt = read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/100dim_multi/gbdt_100dim_real_random_results_dump.json')
reg_gbdt = {k: v['reg'] for k, v in gbdt.items()}
set_gbdt = {k: v['set'] for k, v in gbdt.items()}

model2res = {'GBT': reg_gbdt,
             'LSTM': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/100dim_multi/rnn_100dim_multi_results_dump.json'),
             #'pi-SGD': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/rnn_jannossy_100dim_multi_results_dump.json'),
             'DeepSets (mean)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/100dim_multi/deepsets_100dim_multi_results_dump.json'),
             'DeepSets (sum)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/deepsets_sum_100dim_multi_results_dump.json'),
             'DeepSets (max)': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/2103/deepsets_max_100dim_multi_results_dump.json'),
             'GBeST': set_gbdt}
             # 'DDS_max': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/DDS/dds_max_100dim_multi_results_dump.json'),
             # 'DDS_sum': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/DDS/dds_sum_100dim_multi_results_dump.json'),
             # 'SIRE': read_json('/home/royhir/projects/SetTrees/eval/first_quadrant/outputs/SIRE/rnn_sire_100dim_multi_results_dump.json')}
plot_res(model2res, name2color, name2marker, title='Q1 100dims', scale=1, loc='lower left')
