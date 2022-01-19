import numpy as np
from sklearn import tree
import os
from collections import defaultdict
import pydotplus
import matplotlib.pyplot as plt

from settree.set_tree import Leaf, SetSplitNode

########################################################################################################################
# Dot graph plotting
########################################################################################################################

def get_dot_graph(dt, features_list=None):
    """
    TODO:  Improve the plot of leaf

    Create a dot file for the DT visualization using GraphViz
    https://github.com/lucksd356/DecisionTrees/blob/master/dtree.py

    Parameters
    ----------
    dt : SetSplitNode
         The decision tree root
    features_list: list, optional
                   Ordered list of the features names

    Returns
    -------
    dot_file : string
    """
    nodes = defaultdict(list)

    def document_node(split_num, dt, branch, parent="null", indent=''):
        # if Leaf node - document and exit
        if isinstance(dt, Leaf):
            nodes[split_num].append(['leaf',
                                     str(dt.value),
                                     parent,
                                     branch,
                                     0.0,
                                     dt.weighted_n_node_samples])
            return

        # if a SetNode - continue branch
        else:
            if features_list != None:
                if dt.feature in features_list:
                    feat = features_list[dt.feature]
            else:
                feat = "F" + str(dt.feature)

            if isinstance(dt.threshold, int) or isinstance(dt.threshold, float):
                decision = '{} ({}) >= {:.4f}'.format(feat, dt.op.__name__.lstrip('a'), dt.threshold)
            else:
                decision = '{} ({}) == {:.4f}'.format(feat, dt.op.__name__.lstrip('a'), dt.threshold)

            if dt.use_attention_set != False:
                decision = decision + ' [AS={}]'.format(dt.use_attention_set)

            if dt.use_attention_set_comp:
                decision = decision + ' [Comp]'

            document_node(split_num + 1, dt.left, True, decision, indent + '\t\t')
            document_node(split_num + 1, dt.right, False, decision, indent + '\t\t')
            nodes[split_num].append([split_num + 1, decision, parent, branch, '{:.3f}'.format(dt.impurity), dt.weighted_n_node_samples])
            return

    document_node(0, dt, None)
    dot_file = ['digraph Tree {',
                'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
                'edge [fontname=helvetica] ;'
                ]

    direct_dict = {}
    node_idx = 0
    for nsplit in range(len(nodes)):
        nodes_in_split = nodes[nsplit]
        for node in nodes_in_split:
            split_num, decision, parent, branch, gain, n_samples = node
            if type(split_num) == int:
                direct_dict['%d-%s' % (split_num, decision)] = node_idx
                dot_file.append('%d [label=<%s<br/>gain %s<br/>samples %s>, fillcolor="#e5813900"] ;' % (node_idx,
                                                                                                         decision.replace(
                                                                                                         '>=',
                                                                                                         '&ge;').replace(
                                                                                                         '?', ''),
                                                                                                         gain,
                                                                                                         n_samples))
            else:
                dot_file.append('%d [label=<gain %s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;' % (node_idx,
                                                                                                               gain,
                                                                                                               n_samples,
                                                                                                                decision))

            if parent != 'null':
                if branch:
                    angle = '45'
                    head_label = 'False'
                else:
                    angle = '-45'
                    head_label = 'True'
                p_node = direct_dict['%d-%s' % (nsplit, parent)]

                if nsplit == 1:
                    dot_file.append('%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;' % (p_node, node_idx, angle,head_label))
                else:
                    dot_file.append('%d -> %d ;' % (p_node, node_idx))
            node_idx += 1

    dot_file.append('}')
    dot_data = '\n'.join(dot_file)
    return dot_data


def save_dt_plot(dt, features_list=None, dir='', file_name='dt_graph.jpg'):
    dot_data = get_dot_graph(dt.tree_, features_list)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(os.path.join(dir, file_name))


def save_dt_dot_file(dt, features_list=None, dir='', file_name='dt_graph.txt'):
    dot_data = get_dot_graph(dt.tree_, features_list)
    with open(os.path.join(dir, file_name), "w+") as f:
        f.write(dot_data)


def dot_file_txt_to_png(dot_txt_file, dir='', file_name='dt_graph.jpg'):
    with open(dot_txt_file, "r") as f:
        dot_data = f.readlines()
    graph = pydotplus.graph_from_dot_data(''.join(dot_data))
    graph.write_png(os.path.join(dir, file_name))


def save_sklearn_dt_plot(sklearn_dt, features_list=None, class_names=None, dir='', file_name='dt_graph.jpg'):
    dot_data = tree.export_graphviz(sklearn_dt,
                                    feature_names=features_list,
                                    class_names=class_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(os.path.join(dir, file_name))

########################################################################################################################
# Explainability plotting
########################################################################################################################

def plot_decision_path_matrix(decision_path, X_set):
    record = X_set.records[0]
    n_items = len(record)
    len_path = len(decision_path)

    nodes_strs = [v[0] for v in decision_path]
    nodes_as = [v[1] for v in decision_path]

    r = np.ones((n_items, len_path))
    for j, a_s in zip(range(len_path), nodes_as):
        for i in a_s:
            r[i, j] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # plt.suptitle('Record decision path',y=0.98)

    ax1.imshow(r, interpolation='none', cmap='gray', vmin=0, vmax=1)

    ax1.set_xticklabels([''] + nodes_strs, rotation=80)
    # ax1.set_xlabel('Node description')
    ax1.set_ylabel('Item serial number')

    m = r.mean(1).reshape(-1, 1)
    ax2.imshow(m, interpolation='none', cmap='gray', vmin=0, vmax=1)
    for i in range(n_items):
        ax2.text(0, i, round(1. - m[i].item(),2), ha="center", va="center", color="w")

    ax2.set(xticks=[], yticks=[])
    ax2.set_aspect(1)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Items importance")
    fig.subplots_adjust(left=-0.6)

    plt.show()


def plot_explainability_dict(d, record=None, sort=True, x_label='Items sn', y_label='Count', title='', gt_important_items=[]):

    if np.any(record) != None:
        for i in range(len(record)):
            if i not in d:
                d[i] = 0

    if sort:
        sorted_kv = sorted(d.items(), key=lambda kv: kv[1])[::-1]
        keys = [kv[0] for kv in sorted_kv]
        vals = [kv[1] for kv in sorted_kv]
    else:
        keys = list(d.keys())
        vals = list(d.values())


    x = np.arange(len(keys))

    if len(gt_important_items):
        max_val = max(vals)
        deltas = np.zeros_like(vals)
        for item in gt_important_items:
            deltas[item] = max_val
        plt.bar(x, deltas, alpha=0.5, label='gt important items')
        plt.bar(x, vals, alpha=0.5, label='attention')
        plt.legend()

    else:
        plt.bar(x, vals)

    plt.xticks(x, keys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def points_scatter_plot_explainability_dict(record, d):
    # d - needs to be with keys: items
    min_count = 1
    points_with_count = {'x': [], 'y': [], 's': []}
    points_without_count = {'x': [], 'y': [], 's': []}
    for num, point in enumerate(record):
        if num not in d:
            count = 0
            points_without_count['x'].append(point[0])
            points_without_count['y'].append(point[1])
            points_without_count['s'].append(20*4**count)
        else:
            count = d[num] / min_count
            points_with_count['x'].append(point[0])
            points_with_count['y'].append(point[1])
            points_with_count['s'].append(20*4**count)

    plt.scatter(points_with_count['x'],
                points_with_count['y'],
                s=points_with_count['s'],
                label='points in as')
    plt.scatter(points_without_count['x'],
                points_without_count['y'],
                s=points_without_count['s'],
                label='points not in as')

    plt.hlines(0, -1, 1, colors='k', linestyles=':')
    plt.vlines(0, -1, 1, colors='k', linestyles=':')
    plt.xlabel('X')
    plt.ylabel('Y')
    lgnd = plt.legend(loc="top right", scatterpoints=1, fontsize=10)
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]

    plt.show()


def plot_2d_points_set(set):
    plt.scatter(set[:, 0], set[:, 1])
    plt.hlines(0, -1, 1, colors='k', linestyles=':')
    plt.vlines(0, -1, 1, colors='k', linestyles=':')
    plt.show()
