import numpy as np


def get_attention_set_memory_from_set_tree(tree, set_sample, method='unique'):
    """
    methods:
        unique - count unique combinations of as
        unique_weighted - weighted count of as combinations (weighted by the n_samples)
        item - count the frequency of each item
        item_weighted - weighted sun of each item's frequency
    """

    d = {}
    nodes, attention_set_memory = tree.detailed_decision_path(set_sample)[0]
    for node, as_key in zip(nodes, attention_set_memory):

        if method == 'unique':
            as_key = tuple(as_key)
            if as_key not in d:
                d[as_key] = 0
            d[as_key] += 1

        elif method == 'unique_weighted':
            as_key = tuple(as_key)
            if as_key not in d:
                d[as_key] = 0
            d[as_key] += node['n_samples']

        elif method == 'item':
            if len(as_key):
                for item_key in as_key:
                    if item_key not in d:
                        d[item_key] = 0
                    d[item_key] += 1

        elif method == 'item_weighted':
            if len(as_key):
                for item_key in as_key:
                    if item_key not in d:
                        d[item_key] = 0
                    d[item_key] += node['n_samples']

    return d


def attention_set_memory_dict_to_rank(memory_dict, method='average', output='list'):
    '''
    methods:
        vanilla - assign unique rank per point, naive approach
        average - mean rank (i+j) / 2 to all points with the same count
    output:
        dict - return point2rank mapping
        list - return the list of points in ascending order by the rank [0, 1, ....], (the most frequent first)
    '''
    # method 1: same rank i to all points with the same count
    if method == 'vanilla':
        count2rank = {count: rank for rank, count in enumerate(sorted(set(list(memory_dict.values())), reverse=True))}
        point2rank = {point: count2rank[count] for point, count in memory_dict.items()}

    # method 2: mean rank (i+j) / 2 to all points with the same count
    elif method == 'average':
        s = sorted(memory_dict.items(), key=lambda kv: kv[1], reverse=True)
        count2rank = {}
        for rank, item in enumerate(s):
            if item[1] not in count2rank:
                count2rank[item[1]] = []
            count2rank[item[1]].append(rank)
        count2rank = {k: np.array(v).mean() for k, v in count2rank.items()}
        point2rank = {point: count2rank[count] for point, count in memory_dict.items()}

    if output == 'list':
        # lower rank is better - more frequent
        return [k for k, v in sorted(point2rank.items(), key=lambda item: item[1])]

    else:
        return point2rank


def get_item2rank_from_tree(tree, ds_record):
    # at : a trained attention tree model
    # record : an input set nxd np.array

    # get a list of a tuple of attention set for every node the record traversed through
    # each tuple contains the items indexes that are in the current level AS
    nodes, attention_set_memory = tree.detailed_decision_path(ds_record)[0]

    # count the number of times each item appear in the attention sets list
    valid_as_memory = [tuple(level) for level in attention_set_memory if len(level)]
    if len(valid_as_memory):
        bincount = np.bincount(np.concatenate(valid_as_memory))
    else:
        bincount = np.zeros((len(ds_record.records[0]),))
    item2count = {item_sn: count for item_sn, count in enumerate(bincount)}

    # report item2rank where rank 0 is the most frequent item
    sorted_item2count = sorted(item2count.items(), key=lambda kv: kv[1], reverse=True)

    count2ranks = {}
    for rank, item_count in enumerate(sorted_item2count):
        if item_count[1] not in count2ranks:
            count2ranks[item_count[1]] = []
        count2ranks[item_count[1]].append(rank)
    count2mean_rank = {k: np.array(v).mean() for k, v in count2ranks.items()}
    item2rank = {item: count2mean_rank[count] for item, count in item2count.items()}
    return item2rank


def get_item2rank_from_gbest(gbest, ds_record):
    # gbat : a trained ensemble of attention trees
    # record : an input set nxd np.array

    # get the terminal leafs absolute values for every tree in the ensemble
    # normalize them to get the relative 'importance'per tree
    values = []
    for tree in gbest.estimators_.flatten():
        values.append(np.abs(tree.predict(ds_record).item()))
    values_sum = float(sum(values))
    values = [v / values_sum for v in values]

    # get item2rank mapping for every tree
    item2rank = {i: 0 for i in range(len(ds_record.records[0]))}
    for tree, value in zip(gbest.estimators_.flatten(), values):
        tree_item2rank = get_item2rank_from_tree(tree, ds_record)

        # weighted sum by the tree's importence (normalized terminal leaf value)
        for item, rank in tree_item2rank.items():
            item2rank[item] += value * 2**(-rank)

    return item2rank


def simple_item2importance_score_gb(gb, set_sample):
    # output: point2score mapping, the higher the better
    d = {}
    values = []
    for tree in gb.estimators_.flatten():
        values.append(np.abs(tree.predict(set_sample).item()))
    values_sum = float(sum(values))
    values = [v / values_sum for v in values]

    for tree, value in zip(gb.estimators_.flatten(), values):
        tree_d = get_attention_set_memory_from_set_tree(tree, set_sample, method='item')
        point2rank = attention_set_memory_dict_to_rank(tree_d, method='average', output='dict')

        for p, r in point2rank.items():
            if p not in d:
                d[p] = 0
            d[p] += value * 2**(-r)
    return d


def simple_tuple2importance_score_gb(gb, set_sample):
    # output: point2score mapping, the higher the better
    d = {}
    values = []
    for tree in gb.estimators_.flatten():
        values.append(np.abs(tree.predict(set_sample).item()))
    values_sum = float(sum(values))
    values = [v / values_sum for v in values]

    for tree, value in zip(gb.estimators_.flatten(), values):
        tree_d = get_attention_set_memory_from_set_tree(tree, set_sample, method='unique')
        point2rank = attention_set_memory_dict_to_rank(tree_d, method='average', output='dict')

        for p, r in point2rank.items():
            if p not in d:
                d[p] = 0
            d[p] += value * 2**(-r)
    return d

