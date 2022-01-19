import numpy as np
import math
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from settree.set_data import apply_operation


DTYPE = np.float32
MAX_INT = np.iinfo(np.int32).max


def complementary(curr_as, record):
    return [i for i in range(len(record)) if i not in curr_as]


def mse(y, weights=None):
    """
    Mean squared error for decision tree (ie., mean) predictions
    """
    return np.average((y - np.average(y, weights=weights)) ** 2, weights=weights)


def entropy(y, weights=None):
    """
    Entropy of a label sequence
    """
    hist = np.bincount(y, weights)
    return stats.entropy(hist)
    # ps = hist / np.sum(hist)
    # return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y, weights=None):
    """
    Gini impurity (local entropy) of a label sequence
    """
    hist = np.bincount(y, weights)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])


class DatasetMemoryQueue():
    """
    FIFO queue for keeping running memory of calculated statistics.
    Is used for avoiding excess computations and increase efficiency.
    """

    def __init__(self, operations, use_attention_set, use_attention_set_comp, attention_set_limit):

        """

        The running memory queue is kept as a list of np.arrays with the structure : [flat_base, flat_as_1, ... flat_as_n]
        Where flat_as_i is an array that contains the aggregated results for the ith stage.

        Parameters
        ----------

        operations : list of numpy functions
            The list of aggregation operators to use when building the tree.

        use_attention_set : bool
            Whether to use the attention-set mechanism.

        use_attention_set_comp : bool
            Whether to use the attention-set compatibles set.

        attention_set_limit : int
            The number of ancestors to scan while scanning split criteria with the attention-sets.
            When attention_set_limit=1, scans only the direct ancestors.
        """

        self.operations = operations
        self.n_operations = len(self.operations)

        self.use_attention_set = use_attention_set
        self.use_attention_set_comp = use_attention_set_comp
        self.attention_set_limit = attention_set_limit
        self.attention_set_states = [False, True] if self.use_attention_set_comp else [False]

        self.X = None

    def _init(self, X_set, mask_inds):
        """
        Calculate the flattened dataset without any attention set
        """
        self.n_features = X_set.shape[1]
        self.base_len = self.n_features * self.n_operations

        self.max_memory = self.n_features * self.n_operations
        if self.use_attention_set:
            if self.use_attention_set_comp:
                self.max_memory = self.max_memory * (1 + np.abs(self.attention_set_limit) * 2)
            else:
                self.max_memory = self.max_memory * (1 + np.abs(self.attention_set_limit))

        X = []
        for i in mask_inds:
            record = X_set.records[i]
            X.append(np.concatenate([op(record) for op in self.operations]))

        self.X = X
        self.records_len = np.full(shape=(len(mask_inds,)), fill_value=self.base_len).astype(int)

    def _get_flat_record(self, record, attention_set):
        record = record[attention_set]
        if len(record):
            flat_record = []
            for op in self.operations:
                flat_record.append(op(record))
            flat_record = np.concatenate(flat_record)
        else:
            flat_record = np.full(shape=(self.n_features * self.n_operations,), fill_value=np.finfo(np.float32).min)
        return flat_record

    def _calc_flatten_dataset_for_current_attention_set(self, X_set, mask_inds):

        flat_record_len = self.base_len * 2 if self.use_attention_set_comp else self.base_len

        for ind in mask_inds:
            as_mem = X_set.attention_set[ind]
            record = X_set.records[ind]

            prev_as = as_mem[-1]
            flat_record = self._get_flat_record(record, prev_as)

            if self.use_attention_set_comp:
                prev_as = complementary(prev_as, record)
                flat_comp_record = self._get_flat_record(record, prev_as)
                flat_record = np.concatenate([flat_record, flat_comp_record])

            # if the memory if full
            # dump the first memory, shift the rest to the right and insert a new memory
            if self.records_len[ind] >= self.max_memory:
                mem_flat_record = self.X[ind]
                mem_base = mem_flat_record[:self.base_len]
                mem_as = mem_flat_record[self.base_len:-flat_record_len]
                self.X[ind] = np.concatenate([mem_base, flat_record, mem_as])
                self.records_len[ind] = self.max_memory

            # if there is already a flatten attention-set in memory
            # shift the current memory to the right and insert a new memory
            elif self.records_len[ind] > self.base_len:
                mem_flat_record = self.X[ind]
                mem_base = mem_flat_record[:self.base_len]
                mem_as = mem_flat_record[self.base_len:]
                self.X[ind] = np.concatenate([mem_base, flat_record, mem_as])
                self.records_len[ind] += flat_record_len

            else:
                mem_base = self.X[ind]
                self.X[ind] = np.concatenate([mem_base, flat_record])
                self.records_len[ind] += flat_record_len

    def _get_feature_number_to_args(self, cur_depth, apply_attention_set=False):
        ind2args = {}
        c = 0
        for op in self.operations:
            for i in range(self.n_features):
                ind2args[c] = {'feature': i,
                               'op': op,
                               'use_attention_set': False,
                               'use_attention_set_comp': False}
                c += 1

        # if the model is using the attention-sets - add it to mapping
        if apply_attention_set:
            for as_level in list(range(-1, max(-cur_depth, -self.attention_set_limit) - 1, -1)):
                for as_comp in self.attention_set_states:
                    for op in self.operations:
                        for i in range(self.n_features):
                            ind2args[c] = {'feature': i,
                                           'op': op,
                                           'use_attention_set': as_level,
                                           'use_attention_set_comp': as_comp}
                            c += 1

        return ind2args

    def reset(self):
        del self.X
        del self.records_len
        self.X = None
        self.records_len = None

    def enqueue_and_get(self, X_set, mask_inds, cur_depth):
        if cur_depth == 0:
            self._init(X_set, mask_inds)
            ind2args = self._get_feature_number_to_args(cur_depth, False)
        else:
            if self.use_attention_set:
                self._calc_flatten_dataset_for_current_attention_set(X_set, mask_inds)
                ind2args = self._get_feature_number_to_args(cur_depth, True)
            else:
                ind2args = self._get_feature_number_to_args(cur_depth, False)

        X = np.stack([self.X[i] for i in mask_inds])
        return X, ind2args


class BaseSplitter():
    """
    Abstract splitter class
    Splitters are called by tree builders to find the best splits.
    """

    def __init__(self, classifier=True, criterion='entropy', max_features=None, min_samples_leaf=1, random_state=None):
        """
        Parameters
        ----------
        classifier : bool
            Whether the tree used for classification or regression

        criterion : {"gini", "entropy", "mse"}, default="entropy"
            The function to measure the quality of a split.

        max_features : int, float, {"auto", "sqrt", "log2"} or None, default="auto"
            The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                  `int(max_features * n_features)` features are considered at each
                  split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
              `ceil(min_samples_leaf * n_samples)` are the minimum
              number of samples for each node.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : int, RandomState instance or None, default=None
            Used to pick randomly the `max_features` used at each split.
        """

        self.criterion = criterion
        self.classifier = classifier
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features

    def init(self, n_samples, n_features):

        self.n_samples = n_samples
        self.n_features = n_features

        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = math.ceil(self.min_samples_leaf * self.n_features)

        if self.criterion in CRITERIONS:
            self.criterion = CRITERIONS[self.criterion]
        else:
            raise ValueError('Invalid criterion name {}'.format(self.criterion))

        if isinstance(self.max_features, float):
            self.max_features = math.ceil(self.max_features * n_features)
        elif self.max_features == 'auto' or self.max_features == 'sqrt':
            self.max_features = math.ceil(np.sqrt(self.n_features))
        elif self.max_features == 'log2':
            self.max_features = math.ceil(np.log2(self.n_features))
        elif self.max_features == None:
            self.max_features = self.n_features
        else:
            raise ValueError('Invalid value for max_features, {}'.format(self.max_features))

    def end(self):
        pass

    def node_split(self, X_set, y, mask_inds, cur_depth):
        raise NotImplemented


class ThirdPartySplitter(BaseSplitter):
    """
    Provides common API for external (third party) splitters
    """

    def __init__(self, classifier=True, criterion='entropy', max_features=None, min_samples_leaf=None, random_state=None,
                 operations=None, use_attention_set=True, use_attention_set_comp=True, attention_set_limit=1):

        self.SMALLEST = np.finfo(DTYPE).min
        self.LARGEST = np.finfo(DTYPE).max

        super().__init__(classifier, criterion, max_features, min_samples_leaf, random_state)

        self.operations = operations
        self.use_attention_set = use_attention_set
        self.use_attention_set_comp = use_attention_set_comp
        self.attention_set_limit = attention_set_limit

        self.memory_queue = DatasetMemoryQueue(operations=self.operations,
                                               use_attention_set=self.use_attention_set,
                                               use_attention_set_comp=self.use_attention_set_comp,
                                               attention_set_limit=self.attention_set_limit)

    def init(self, n_samples, n_features):
        self.attention_set_states = [False, True] if self.use_attention_set_comp else [False]
        self.n_samples = n_samples
        self.n_features = n_features * len(self.operations)
        if self.use_attention_set:
            n_attention_set_levels = np.abs(self.use_attention_set)
            if self.use_attention_set_comp:
                self.n_features = self.n_features * (n_attention_set_levels * 2 + 1)
            else:
                self.n_features = self.n_features * (n_attention_set_levels + 1)

        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = math.ceil(self.min_samples_leaf * self.n_features)

        if self.criterion in CRITERIONS:
            self.criterion = CRITERIONS[self.criterion]
        else:
            raise ValueError('Invalid criterion name {}'.format(self.criterion))

    def node_split(self, X_set, y, mask_inds, cur_depth):
        raise NotImplemented

    def end(self):
        self.memory_queue.reset()


class SklearnSetSplitter(ThirdPartySplitter):

    def __init__(self, classifier=True, criterion='entropy', max_features=None, min_samples_leaf=None, random_state=1,
                 operations=None, use_attention_set=False, use_attention_set_comp=True, attention_set_limit=1):

        super().__init__(classifier, criterion, max_features, min_samples_leaf, random_state,
                         operations, use_attention_set, use_attention_set_comp, attention_set_limit)

    def node_split(self, X_set, y, mask_inds, cur_depth):

        y = y.take(mask_inds)
        X, ind2args = self.memory_queue.enqueue_and_get(X_set, mask_inds, cur_depth)
        rand_state = self.random_state.randint(0, MAX_INT)

        if self.classifier:
            clf = DecisionTreeClassifier(criterion='entropy',
                                         splitter='best',
                                         max_depth=1,
                                         min_samples_split=2,
                                         min_samples_leaf=1,
                                         min_weight_fraction_leaf=0.0,
                                         max_features=self.max_features,
                                         random_state=rand_state).fit(X, y)
        else:
            clf = DecisionTreeRegressor(criterion='mse',
                                        splitter='best',
                                        max_depth=1,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        min_weight_fraction_leaf=0.0,
                                        max_features=self.max_features,
                                        random_state=rand_state).fit(X, y)

        threshold = clf.tree_.threshold[0]
        feature_num = clf.tree_.feature[0]
        impurity = clf.tree_.impurity[0]

        if feature_num < 0:
            feature_num = 0
            is_leaf = True
        else:
            is_leaf = False

        best_split = ind2args[feature_num]
        best_split['threshold'] = threshold
        best_split['impurity'] = impurity
        best_split['is_leaf'] = is_leaf
        return best_split


SPLITTERS = {'sklearn': SklearnSetSplitter}
CRITERIONS = {'gini': gini, 'entropy': entropy, 'mse': mse}
