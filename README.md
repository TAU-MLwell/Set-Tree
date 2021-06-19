# Set-Tree
### Extending decision trees to process sets
This is the official repository for the paper: "Trees with Attention for Set Prediction Tasks" (ICML21).

This repository contains a prototypical implementaion of Set-Tree and GBeST (Gradient Boosted Set-Tree) algorithms

## Getting Started
The Set-Tree package can be downloaded from PIP:
`pip install settree`

We also supply the code and datasets for reproducing our experimetns under `exps` folder.

## Background and motivation
In many machine learning applications, each record represents a set of items. A set is an unordered group of items, the number of items may differ between different sets. Problems comprised from sets of items are present in diverse fields, from particle physics and cosmology to statistics and computer graphics. In this work, we present a novel tree-based algorithm for processing sets.

![set_problems](images/set_problems.PNG)

## The model
Set-Tree model comprised from two components:
1) **Set-compatible split creteria**: we specifically support the familly of split creteria defined by the following equation and parametrized by alpha and beta.
2) **Attention-Sets**: a mechanism for allplying the split creteria to subsets of the input. The attention-sets are derived forn previous split-creteria and allows the model to learn more complex set-functions.


<img src="https://github.com/TAU-MLwell/Set-Tree/blob/main/images/model.PNG" width="600" align="center">

## Implementation

The current implementation is based on Sklean's `BaseEstimator` class and is fully compatible with Sklearn.
It contains two main components: `SetDataset` object, that receives `records` as attribute. `records` is a list of numpy arrays, each array with shape `(n_i, d)` represents a single record (set). `d` is the dimention of each item in the record and `n_i` is the number of items in the i's record and may differ between records. The second componnent is `SetTree` model inherited from Sklean's `BaseEstimator` and has simillar attributes.

When configuring Set-Tree one should also configure:
- `operations` : list of the operations to be used
- `use_attention_set` : binary flag for activating the attention-sets mechanism 
- `attention_set_limit` :  the number of ancestors levels to derive attention-sets from
- `use_attention_set_comp` : binary flag for activating the attention-sets compatibility option

A simplified code snippet:
```
import settree
import numpy as np

set_data = settree.SetDataset(records=[np.random.randn(2,5) for _ in range(10)])
labels = np.random.randn(10) >= 0.5
set_tree_model = settree.SetTree(classifier=True,
                                 criterion='entropy',
                                 operations=settree.OPERATIONS,
                                 use_attention_set=True,
                                 use_attention_set_comp=True,
                                 attention_set_limit=5,
                                 max_depth=10)
set_tree_model.fit(set_data, labels)
```
For further details and examples see: `example.ipynb`.

## Citation
If you use Set-Tree in your work, please cite:

## License
Set-Tree is MIT licensed, as found in the [LICENSE](https://github.com/TAU-MLwell/Set-Tree/blob/main/LICENSE) file.







