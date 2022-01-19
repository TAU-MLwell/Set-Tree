import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import settree


# Data params
SET_SIZE = 5
ITEM_DIM = 5
N_TRAIN = 1000
N_TEST = 100

# x_train, y_train = settree.get_first_quarter_data(N_TRAIN, min_items_set=SET_SIZE, max_items_set=SET_SIZE+1, dim=ITEM_DIM)
# x_test, y_test = settree.get_first_quarter_data(N_TEST, min_items_set=SET_SIZE, max_items_set=SET_SIZE+1, dim=ITEM_DIM)
# ds_train = settree.SetDataset(records=x_train, is_init=True)
# ds_test = settree.SetDataset(records=x_test, is_init=True)
#
# print('Train dataset object: ' + str(ds_train))
# print('Test dataset object: ' + str(ds_test))


list_of_operations = settree.OPERATIONS
import settree
import numpy as np

# set_data = settree.SetDataset(records=[np.random.randn(6, 5) for _ in range(1000)])
# labels = np.random.randn(1000) >= 0.5
# gbest_model = settree.GradientBoostedSetTreeClassifier(learning_rate=0.1,
#                                                        n_estimators=3,
#                                                        criterion='mse',
#                                                        operations=settree.OPERATIONS,
#                                                        use_attention_set=True,
#                                                        use_attention_set_comp=True,
#                                                        attention_set_limit=5,
#                                                        max_depth=10)
# gbest_model.fit(set_data, labels)
# print(gbest_model.feature_importances_)


records = [np.random.randn(1, 5) for _ in range(100)]
mod_records = []
for sn, i in enumerate(records):
    if sn < 50:
        i[0, 0] = np.random.randn() * 6
    mod_records.append(i)
set_data = settree.SetDataset(records=mod_records)
labels = np.array([0] * 50 + [1] * 50)
gbest_model = settree.GradientBoostedSetTreeClassifier(learning_rate=0.1,
                                                       n_estimators=30,
                                                       criterion='mse',
                                                       operations=settree.OPERATIONS,
                                                       use_attention_set=True,
                                                       use_attention_set_comp=True,
                                                       attention_set_limit=5,
                                                       max_depth=10)
gbest_model.fit(set_data, labels)
print(gbest_model.feature_importances_)