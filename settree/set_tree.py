import numpy as np
import math
import queue

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from settree.set_data import SetDataset, OPERATIONS, split_set_dataset, apply_operation, update_attention_set, mask_x_set
from settree.splitters import SPLITTERS


class SetSplitNode:
    def __init__(self, left, right, n_samples, depth, criteria_args):
        self.left = left
        self.right = right
        self.weighted_n_node_samples = n_samples
        self.depth = depth

        self.impurity = criteria_args['impurity']
        self.use_attention_set = criteria_args['use_attention_set']
        self.use_attention_set_comp = criteria_args['use_attention_set_comp']

        self.op = criteria_args['op']
        self.feature = criteria_args['feature']
        self.threshold = criteria_args['threshold']

    def __repr__(self):
        s = self.__class__.__name__
        s += " (Feature {} ({}) > {:.5f})".format(self.feature, self.op.__name__, self.threshold)
        return s

    @property
    def str(self):
        if self.use_attention_set:
            return "(F{} ({}) > {:.4f} [{}])".format(self.feature, self.op.__name__, self.threshold,
                                                     self.use_attention_set, ' [C]' if self.use_attention_set_comp else '')
        else:
            return "(F{} ({}) > {:.4f})".format(self.feature, self.op.__name__, self.threshold)


class Leaf:
    def __init__(self, value, n_samples, sn, depth=0):
        self.value = value
        self.weighted_n_node_samples = n_samples
        self.depth = depth
        self.sn = sn

    def __repr__(self):
        s = self.__class__.__name__
        s += " (value: {}, n_samples: {})".format(self.value, self.weighted_n_node_samples)
        return s


class SetTree(BaseEstimator):
    """
    SetTree base class
    Inherited from sklearn's BaseEstimator
    """

    def __init__(self, classifier=True, criterion='entropy', splitter='sklearn',
                 max_features=None, min_samples_split=2,
                 operations=OPERATIONS, use_attention_set=True,
                 use_attention_set_comp=True, attention_set_limit=1,
                 max_depth=None, min_samples_leaf=None, random_state=None):

        """
        Parameters
        ----------
        classifier : bool
            Whether the tree used for classification or regression

        criterion : {"gini", "entropy", "mse"}, default="entropy"
            The function to measure the quality of a split.

        splitter : {"set", "xgboost", "sklearn"}, default="sklearn"
            The current version is using an external engine for finding the optimal split.

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

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

        operations : list of numpy functions
            The list of aggregation operators to use when building the tree.

        use_attention_set : bool, default=True
            Whether to use the attention-set mechanism.

        use_attention_set_comp : bool, default=True
            Whether to use the attention-set compatibles set.

        attention_set_limit : int, default=1
            The number of ancestors to scan while scanning split criteria with the attention-sets.
            When attention_set_limit=1, scans only the direct ancestors.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

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

        random_state : int, RandomState instance or None, default=None
            Used to pick randomly the `max_features` used at each split.
        """

        self.classifier = classifier
        self.criterion = criterion
        self.max_features = max_features
        self.min_samples_split = min_samples_split

        self.splitter = splitter
        self.operations = operations
        self.use_attention_set = use_attention_set
        self.use_attention_set_comp = use_attention_set_comp
        self.attention_set_limit = attention_set_limit

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def init(self, n_classes, n_samples, n_features):
        random_state = check_random_state(self.random_state)

        self.max_depth_ = self.max_depth if self.max_depth else np.inf
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_features = n_features

        self.n_leafs = 0
        self.tree_ = None
        self.leafs = []
        self.depth = 0

        if self.splitter in SPLITTERS:
            self.splitter_ = SPLITTERS[self.splitter](classifier=self.classifier,
                                                      criterion=self.criterion,
                                                      max_features=self.max_features,
                                                      min_samples_leaf=self.min_samples_leaf,
                                                      random_state=random_state,
                                                      operations=self.operations,
                                                      use_attention_set=self.use_attention_set,
                                                      use_attention_set_comp=self.use_attention_set_comp,
                                                      attention_set_limit=self.attention_set_limit)
        else:
            raise ValueError('Invalid splitter name: {}'.format(self.splitter))


        if isinstance(self.min_samples_split, float):
            self.min_samples_split = math.ceil(self.min_samples_split * self.n_features)

    def _get_node(self, y, mask_inds, criteria_args, cur_depth):

        def __add_leaf(val, n_samples, curr_depth):
            l = Leaf(val, n_samples=n_samples, sn=self.n_leafs, depth=curr_depth)
            self.n_leafs += 1
            self.leafs.append(l)
            return l

        y = y.take(mask_inds)
        n_samples = len(y)

        # if all labels are the same, return a leaf
        if len(set(y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[y[0]] = 1.0
                return __add_leaf(prob, n_samples, cur_depth)
            else:
                return __add_leaf(y[0], n_samples, cur_depth)

        # if reached max_depth, or the number of samples is under a threshold
        # or criteria_args is explicitly a leaf (from ThirdPartySplitter) -> the return a leaf
        if cur_depth >= self.max_depth_ or\
           n_samples < self.min_samples_split or\
           criteria_args.get('is_leaf', False):

            if self.classifier:
                v = np.bincount(y, minlength=self.n_classes) / n_samples
            else:
                v = np.mean(y, axis=0)
            return __add_leaf(v, n_samples, cur_depth)

        return SetSplitNode(None, None, n_samples, cur_depth, criteria_args)

    def fit(self, X_set, y, sample_weight=None):
        n_classes = int(max(y)) + 1 if self.classifier else None
        n_samples, n_features = X_set.shape

        # init the tree building process
        self.init(n_classes, n_samples, n_features)
        # init the splitter
        self.splitter_.init(n_samples, n_features)
        self.tree_ = self._build(X_set, y, sample_weight)
        self.splitter_.end()
        return self

    def _build(self, X_set, y, sample_weight=None):
        mask_inds = list(range(len(X_set)))
        split_args = self.splitter_.node_split(X_set, y, mask_inds, 0)
        root = self._get_node(y, mask_inds, split_args, 0)
        mask_inds_l, mask_inds_r = split_set_dataset(X_set, mask_inds, split_args)

        nodes = queue.Queue()
        datas = queue.Queue()

        if isinstance(root, SetSplitNode):
            nodes.put(root)
            datas.put(mask_inds_l)
            datas.put(mask_inds_r)

        while not nodes.empty():
            node = nodes.get()

            # expend to left subtree
            mask_inds_l = datas.get()
            left_split_args = self.splitter_.node_split(X_set, y,  mask_inds_l, node.depth + 1)
            left_node = self._get_node(y, mask_inds_l, left_split_args, node.depth + 1)
            node.left = left_node
            if isinstance(left_node, SetSplitNode):
                mask_inds_l, mask_inds_r = split_set_dataset(X_set, mask_inds_l, left_split_args)
                if not len(mask_inds_l) or not len(mask_inds_r):
                    print_debug_error(mask_inds_l, mask_inds_r, y, left_split_args)

                datas.put(mask_inds_l)
                datas.put(mask_inds_r)
                nodes.put(left_node)

            # expend to right subtree
            mask_inds_r = datas.get()
            right_split_args = self.splitter_.node_split(X_set, y, mask_inds_r, node.depth + 1)
            right_node = self._get_node(y, mask_inds_r, right_split_args, node.depth + 1)
            node.right = right_node

            if isinstance(right_node, SetSplitNode):
                mask_inds_l, mask_inds_r = split_set_dataset(X_set, mask_inds_r, right_split_args)
                if not len(mask_inds_l) or not len(mask_inds_r):
                    print_debug_error(mask_inds_l, mask_inds_r, y, right_split_args)

                datas.put(mask_inds_l)
                datas.put(mask_inds_r)
                nodes.put(right_node)

            max_depth = max(right_node.depth, left_node.depth)
            if max_depth > self.depth:
                self.depth = max_depth

        return root

    def _traverse(self, X_set, node):

        # if reached a leaf - exit
        while not isinstance(node, Leaf):
            X, _, _ = apply_operation(X_set=X_set,
                                      mask_inds=[0],
                                      op=node.op,
                                      attention_set_level=node.use_attention_set,
                                      attention_set_comp=node.use_attention_set_comp)

            # record the current's node attention set
            update_attention_set(X_set=X_set,
                                 mask_inds=[0],
                                 op=node.op,
                                 feat=node.feature,
                                 thresh=node.threshold,
                                 attention_set_level=node.use_attention_set,
                                 attention_set_comp=node.use_attention_set_comp)

            # if empty - turn left
            if not len(X):
                node = node.left
            else:
                if X[0][node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node

    def _traverse_with_memory(self, X_set, node, nodes_out=False):

        # if reached a leaf - exit
        nodes = []
        while not isinstance(node, Leaf):
            nodes.append(node.__dict__)

            X, _, _ = apply_operation(X_set=X_set,
                                      mask_inds=[0],
                                      op=node.op,
                                      attention_set_level=node.use_attention_set,
                                      attention_set_comp=node.use_attention_set_comp)

            # record the current's node attention set
            update_attention_set(X_set=X_set,
                                 mask_inds=[0],
                                 op=node.op,
                                 feat=node.feature,
                                 thresh=node.threshold,
                                 attention_set_level=node.use_attention_set,
                                 attention_set_comp=node.use_attention_set_comp)

            # if empty - turn left
            if not len(X):
                node = node.left
            else:
                if X[0][node.feature] < node.threshold:
                    node = node.left
                else:
                    node = node.right

        if nodes_out:
            return nodes, X_set.attention_set[0]
        else:
            return X_set.attention_set[0]

    def _get_leaf_val(self, X_set, node, prob=False):
        node = self._traverse(X_set, node)
        # whet reached a leaf - report its value
        if self.classifier:
            return node.value if prob else node.value.argmax()
        return node.value

    def _get_leaf_ind(self, X_set, node):
        node = self._traverse(X_set, node)
        return node.sn

    def _get_leaf_val_and_ind(self, X_set, node, prob=False):
        node = self._traverse(X_set, node)
        # whet reached a leaf - report its value
        if self.classifier:
            return node.value if prob else node.value.argmax()
        return node.value, node.sn

    def compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""

        def get_all_nodes(root, mem):
            if isinstance(root.right, SetSplitNode) and isinstance(root.left, SetSplitNode):
                mem.append(root)
            if isinstance(root.right, SetSplitNode):
                get_all_nodes(root.right, mem)
            if isinstance(root.left, SetSplitNode):
                get_all_nodes(root.left, mem)
            return mem

        importances = np.zeros((self.n_features,))
        nodes = get_all_nodes(self.tree_, mem=[])
        for split_node in nodes:
                importances[split_node.feature] += (split_node.weighted_n_node_samples * split_node.impurity -
                                                    split_node.left.weighted_n_node_samples * split_node.left.impurity -
                                                    split_node.right.weighted_n_node_samples * split_node.right.impurity)

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances


    def predict(self, X_set):
        return np.array([self._get_leaf_val(SetDataset(records=[x]), self.tree_) for x in X_set.records])

    def apply(self, X_set):
        return np.array([self._get_leaf_ind(SetDataset(records=[x]), self.tree_) for x in X_set.records])

    def predict_and_apply(self, X_set):
        preds_and_inds = [self._get_leaf_val_and_ind(SetDataset(records=[x]), self.tree_) for x in X_set.records]
        return np.array([i[0] for i in preds_and_inds]), np.array([i[1] for i in preds_and_inds])

    def predict_proba(self, X_set):
        if not self.classifier:
            raise ValueError("predict_probs` undefined for classifier = False")
        return np.array([self._get_leaf_val(SetDataset(records=[x]), self.tree_, prob=True) for x in X_set.records])

    def _get_decision_path(self, X_set, node):
        m = self._traverse_with_memory(X_set, node, nodes_out=False)
        return m

    def _get_detailed_decision_path(self, X_set, node):
        nodes, m = self._traverse_with_memory(X_set, node, nodes_out=True)
        return nodes, m

    def decision_path(self, X_set):
        return [self._get_decision_path(SetDataset(records=[x]), self.tree_) for x in X_set.records]

    def detailed_decision_path(self, X_set):
        return [self._get_detailed_decision_path(SetDataset(records=[x]), self.tree_) for x in X_set.records]

    @property
    def n_nodes(self):
        def count_nodes(root):
            if isinstance(root, Leaf):
                return 0

            res = 0
            if (root.left and root.right):
                res += 1
            res += (count_nodes(root.left) +
                    count_nodes(root.right))
            return res
        return count_nodes(self.tree_)


def print_debug_error(mask_inds_l, mask_inds_r, y, args):
    print('There are {} left and {} right inds'.format(len(mask_inds_l), len(mask_inds_r)))

    if not len(mask_inds_l):
        print('Invalid data {} split'.format('left'))
        print('There are {} unique y(right inds) values'.format(len(set(y.take(mask_inds_r)))))
    else:
        print('Invalid data {} split'.format('right'))
        print('There are {} unique y(left inds) values'.format(len(set(y.take(mask_inds_l)))))

    if 'is_leaf' in args:
        print('args["is_leaf"] = {}'.format(args['is_leaf']))

