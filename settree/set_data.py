import random
import numpy as np
import settree.operations as ops

OPERATIONS = [ops.Min(), ops.Max(), ops.Sum(), ops.Mean(), ops.SecondMomentMean(), ops.HarmonicMean(), ops.GeometricMean()]
OPERATIONS_BASIC = [ops.Min(), ops.Max(), ops.Sum(), ops.Mean()]
DTYPE = np.float32


class SetDataset():
    """
    Object for handling sets
    """

    def __init__(self, records, attention_sets_hist=None, is_init=False):
        """
        Parameters
        ----------

        records : list of numpy arrays
            Each cell in the list is a matrix that represents a set (record): [Ni, d]
            where Ni is the number of items in the set and d id the items' dimension.
            Ni may be different for every entry in the list (sets with different cardinallity).

        attention_sets_hist : list of lists, default: None
            Keeps the attention sets history for all the records

        is_init : bool, default: False
            If TRUE, preprocess the records to the current data type

        """

        if is_init:
            if isinstance(records[0], list):
                 records = [np.array(record).astype(DTYPE) for record in records]
            if not isinstance(records[0], DTYPE):
                records = [record.astype(DTYPE) for record in records]

        self.records = records
        self.n_features_ = records[0].shape[1]

        if attention_sets_hist is None:
            self.attention_set = [[] for _ in range(len(self.records))]
        else:
            self.attention_set = attention_sets_hist

    def get_subset(self, inds):
        return SetDataset(records=[self.records[i] for i in inds],
                          attention_sets_hist=[self.attention_set[i] for i in inds])

    def get_masked_subset(self, mask):
        inds = np.where(mask == True)[0]
        return self.get_subset(inds)

    def __len__(self):
        return len(self.records)

    @property
    def shape(self):
        return (self.__len__(), self.n_features)

    @property
    def n_features(self):
        return self.n_features_

    @property
    def records_lens(self):
        return [len(r) for r in self.records]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += "num_records={}, ".format(len(self))
        s += "num_features={})".format(self.records[0].shape[1])
        return s


def complementary(curr_as, record):
    return [i for i in range(len(record)) if i not in curr_as]


def set_object_to_matrix(X_set, operations_list):
    ''' Export a np.matrix with super vector per record
        meaning every record - a set will be flattened to
        a long vector contains all its features under all the operations'''
    X = []
    for r in X_set.records:
        t = []
        for op in operations_list:
            t += op(r).tolist()
        X.append(t)
    return np.array(X)


def flatten_datasets(ds_train, ds_test, operations_list, ds_val=None):
    train_x = set_object_to_matrix(ds_train, operations_list)
    test_x = set_object_to_matrix(ds_test, operations_list)
    if ds_val:
        val_x = set_object_to_matrix(ds_val, operations_list)
        return train_x, test_x, val_x
    else:
        return train_x, test_x


def apply_operation(X_set, mask_inds, op, feature=None, attention_set_level=False, attention_set_comp=False):
    """
    Applies an operation feature-wise for every set element in the data list.
    """
    empty_records_indxs = []
    valid_records_indxs = []

    if attention_set_level:
        valid_records = []
        for ind in mask_inds:
            as_mem = X_set.attention_set[ind]
            record = X_set.records[ind]

            if attention_set_comp:
                prev_as = complementary(as_mem[attention_set_level], record)
            else:
                prev_as = as_mem[attention_set_level]

            record = record[prev_as]
            if len(record):
                valid_records.append(record)
                valid_records_indxs.append(ind)
            else:
                empty_records_indxs.append(ind)

    else:
        valid_records = [X_set.records[i] for i in mask_inds]
        valid_records_indxs = mask_inds
        empty_records_indxs = []

    # all records are empty
    if not len(valid_records):
        return np.empty(0), valid_records_indxs, empty_records_indxs
    # apply op for a single feature
    if feature != None:
        return np.array([op(sample) for sample in valid_records])[:, feature], valid_records_indxs, empty_records_indxs
    # there are some empty records or none is empty
    else:
        return np.stack([op(sample) for sample in valid_records]), valid_records_indxs, empty_records_indxs


def apply_operation_and_pad(X_set, mask_inds, op, attention_set_level=False, attention_set_comp=False):
    """
    Applies an operation feature-wise for every set element in the data list.
    """

    if attention_set_level:
        data = []
        for ind in mask_inds:
            as_mem = X_set.attention_set[ind]
            record = X_set.records[ind]

            prev_as = as_mem[attention_set_level]
            if attention_set_comp:
                prev_as = complementary(prev_as, record)

            record = record[prev_as]
            if len(record):
                data.append(op(record))
            else:
                data.append([np.finfo(np.float32).min] * X_set.shape[1])

    else:
        data = [op(X_set.records[i]) for i in mask_inds]

    return np.stack(data)


def split_set_dataset(X_set, mask_inds, criteria_args):
    """
    Given a certain rul split the data samples to two groups.
    For the positive group rule(sample) = True and for the negative group rule(sample) = False
    Parameters
    """

    opperation_name = criteria_args['op']
    feature = criteria_args['feature']
    threshold = criteria_args['threshold']
    use_attention_set = criteria_args['use_attention_set']
    use_attention_set_comp = criteria_args['use_attention_set_comp']
    X, valid_records_indxs, empty_records_indxs = apply_operation(X_set=X_set,
                                                                  mask_inds=mask_inds,
                                                                  op=opperation_name,
                                                                  feature=feature,
                                                                  attention_set_level=use_attention_set,
                                                                  attention_set_comp=use_attention_set_comp)

    l = [org_ind for val, org_ind in zip(X, valid_records_indxs) if val < threshold] + empty_records_indxs
    r = [org_ind for val, org_ind in zip(X, valid_records_indxs) if val >= threshold]

    update_attention_set(X_set, mask_inds, opperation_name, feature, threshold, use_attention_set, use_attention_set_comp)

    return l, r


def update_attention_set(X_set, mask_inds, op, feat, thresh, attention_set_level, attention_set_comp):

    for ind in mask_inds:
        as_mem = X_set.attention_set[ind]
        record = X_set.records[ind]

        if len(as_mem) and attention_set_level:
            if attention_set_comp:
                prev_as = complementary(as_mem[attention_set_level], record)
            else:
                prev_as = as_mem[attention_set_level]

            # if prev AS is empty, the current will also be empty
            if not len(prev_as):
                current_level_as = prev_as

            else:
                # insert only the items that are valid from the prev attention set
                record = record[prev_as]
                curr2prev = {sn: i for sn, i in enumerate(prev_as)}
                current_level_as = [curr2prev[i] for i in op.get_as(record[:, feat], thresh)]

        # if no prev AS memory
        else:
            current_level_as = op.get_as(record[:, feat], thresh)

        X_set.attention_set[ind].append(current_level_as)


def mask_x_set(X_set, y, mask_inds):
    y_mask = y.take(mask_inds)
    X_mask = [X_set[i] for i in mask_inds]
    return X_mask, y_mask


def merge_init_datasets(ds_a, ds_b):
    records = ds_a.records + ds_b.records
    return SetDataset(records=records, is_init=True)


def get_first_quarter_data(num_samples, min_items_set=2, max_items_set=10, dim=2):

    def inject_samples_in_first_quarter(set_of_samples, min=1, max=1, dim=2):
        num = random.choice(range(min, max + 1))
        pos_points = np.random.uniform(low=0, high=1, size=(num, dim))
        set_of_samples[:num, :] = pos_points
        return set_of_samples

    def sample_point_not_from_first_quarter(dim=2):
        # sample a quarter (not the first)
        while True:
            r = np.random.uniform(-1, 1, dim)
            if sum(r >= 0) < dim:
                break
        return tuple(r)

    def sample_set(num, dim):
        return np.stack([sample_point_not_from_first_quarter(dim) for _ in range(num)])

    s_1 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [sample_set(random.choice(range(min_items_set, max_items_set)), dim) for _ in range(num_samples // 2)]
    s_2 = [inject_samples_in_first_quarter(i, min=1, max=1, dim=dim) for i in s_2]

    x = s_1 + s_2
    y = np.concatenate([np.zeros(len(s_1)), np.ones(len(s_2))]).astype(np.int64)
    return x, y
