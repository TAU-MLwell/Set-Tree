import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import sys
import pickle
import copy

sys.path.append('/home/royhir/projects/SetTrees/')

from settree.set_data import SetDataset, OPERATIONS, OPERATIONS_BASIC
from data.icd9 import ICD9
import exps.eval_utils as eval
import logging
import os

log_dir = os.path.join(os.path.abspath('__file__' + '/../'), 'outputs', 'mild')
exp_name = ''
eval.create_logger(log_dir=log_dir,
                   log_name=exp_name,
                   dump=False)


class ParseICD9code():
    def __init__(self, codes_with_meta):
        self.codes_with_meta = codes_with_meta

    def _internal_parse(self, code):
        if 'V' in code:
                return code[:3] + '.' + code[3:]

        if 'E' in code:
            if len(code) > 4:
                return code[:4] + '.' + code[4:]
            else:
                return code

        code = code.rstrip('0')

        if len(code) == 1:
            return '00' + code + '.0'

        elif len(code) == 2:
            return '0' + code

        elif(code) == 3:
            return code

        elif len(code) > 3:
            if code[3] == '0':
                return code[:3] + '.' + code[4:]
            else:
                return code[:3] + '.' + code[3:]
        else:
            return code

    def parse(self, code):
        parse_code = self._internal_parse(code)
        if parse_code in self.codes_with_meta:
            return parse_code
        elif code in self.codes_with_meta:
            return code

        elif code[:-1] in self.codes_with_meta:
            return code[:-1]
        elif code[:-2] in self.codes_with_meta:
            return code[:-2]
        elif code[1:] in self.codes_with_meta:
            return code[1:]
        elif code[2:] in self.codes_with_meta:
            return code[2:]

        elif parse_code[:-1] in self.codes_with_meta:
            return parse_code[:-1]
        elif parse_code[:-2] in self.codes_with_meta:
            return parse_code[:-2]
        elif parse_code[:-3] in self.codes_with_meta:
            return parse_code[:-3]
        else:
            return parse_code


class SimpleICD9CategoryParser():
    ''' Parse a ICD9 to it's higher level hierarchy '''

    range2level = {(1, 139): 0,
                   (140, 239): 1,
                   (240, 279): 2,
                   (280, 289): 3,
                   (290, 319): 4,
                   (320, 389): 5,
                   (390, 459): 6,
                   (460, 519): 7,
                   (520, 579): 8,
                   (580, 629): 9,
                   (630, 679): 10,
                   (680, 709): 11,
                   (710, 739): 12,
                   (740, 759): 13,
                   (760, 779): 14,
                   (780, 799): 15,
                   (800, 999): 16
                   }

    def parse(self, code):
        if 'V' in code or 'E' in code:
            return 17
        else:
            code_int = int(code[:3])
            for r, level in self.range2level.items():
                if r[0] <= code_int <= r[1]:
                    return level
            print(code)
            raise ValueError

# Load datasets
np.random.seed(42)

pres_df = pd.read_csv('/mnt/drive2/datasets/mimicIII_1.4/PRESCRIPTIONS.csv')
logging.info('Loaded PRESCRIPTIONS table: shape: {}'.format(pres_df.shape))

diag2icd_df = pd.read_csv('/mnt/drive2/datasets/mimicIII_1.4/DIAGNOSES_ICD.csv')
valid_icd9_codes = diag2icd_df['ICD9_CODE'].unique()
valid_icd9_codes = [i for i in valid_icd9_codes if not isinstance(i, float)] # drop nan
logging.info('Loaded DIAGNOSES_ICD table: shape: {} | num unique IDC9 codes {}'.format(diag2icd_df.shape, len(valid_icd9_codes)))
parser = SimpleICD9CategoryParser()
code2category = {i: parser.parse(i) for i in valid_icd9_codes}

def create_data(keys, method='mild'):
    # method: mild / strict

    pos_records = []
    for key in keys:
        pos_records.append(drug_embeds.take([drug2ind[i] for i in valid_key2drugs[key]], axis=0))

    neg_records = []
    for i in range(len(pos_records)):
        if method == 'mild':
            n_neg = np.random.randint(1, min(len(pos_records[i]), 5), 1)[0]
        else:
            n_neg = 1
        neg = copy.deepcopy(pos_records[i])
        neg[:n_neg] = drug_embeds.take(np.random.randint(0, len(valid_drugs), n_neg), 0)
        neg_records.append(neg)

    d = pos_records + neg_records
    y = np.array([1] * len(pos_records) + [0] * len(neg_records)).astype(np.int64)
    return d, y


#%%
# params
PR_GROUPS_TRAIN = 0.8
PR_HAM_TEST = 0.2

MIN_NUM_DRUGS = 2
MAX_NUM_DRUGS = 100

MAX_IDCS_PER_PATIENT = 5

DRUGS_MIN_FREQ = 50
DRUGS_MAX_FREQ = 10000

# get <key : drugs> from PRESCRIPTIONS table
# filter the drugs by frequency
pres_df = pres_df[pres_df['NDC'].notna()]
pres_df = pres_df.astype({'NDC': 'int64'})
pres_df = pres_df.loc[:, ['SUBJECT_ID', 'HADM_ID', 'NDC']]
key2drugs = pres_df.groupby(['SUBJECT_ID', 'HADM_ID'])['NDC'].apply(list).to_dict()
drugs_counter = Counter(np.concatenate(list(key2drugs.values())))
valid_drugs = []
for drug, count in drugs_counter.items():
    if DRUGS_MIN_FREQ < count <= DRUGS_MAX_FREQ:
        valid_drugs.append(drug)
logging.info('There are {} valid drugs (out of {})'.format(len(valid_drugs), len(drugs_counter)))

# filter the valid keys by the number of different drugs
valid_key2drugs = {}
for key, drugs_list in key2drugs.items():
    drugs_list = list(filter(lambda x: x in valid_drugs, list(set(drugs_list))))
    if MIN_NUM_DRUGS <= len(drugs_list) < MAX_NUM_DRUGS:
        valid_key2drugs[key] = drugs_list
valid_keys = list(valid_key2drugs.keys())
logging.info('There are {} valid drug groups'.format(len(valid_key2drugs)))
lens = np.array([len(v) for v in valid_key2drugs.values()])
logging.info('Max: {} min {} mean {:.1f} std {:.1f}'.format(lens.max(), lens.min(), lens.mean(), lens.std()))

# DEBUG
lens = np.array(list(Counter(np.concatenate(list(valid_key2drugs.values()))).values()))
logging.info('After all filtering - the distribution of drugs is:')
logging.info('Max: {} min {} mean {:.1f} std {:.1f}'.format(lens.max(), lens.min(), lens.mean(), lens.std()))

#%%
# get <key : icd ordered list> dict
key2icd_seq = diag2icd_df.groupby(['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].apply(list).to_dict()
valid_key2icd_seq = {k: v for k, v in key2icd_seq.items() if k in valid_keys}
logging.info('There are {} keys with valid ICD9 seq (out of {})'.format(len(valid_key2icd_seq), len(key2icd_seq)))

for seed in range(5):
    logging.info('Seed {}'.format(seed))
    # split to train/test by subjects
    valid_subjects = list(set([i[0] for i in valid_keys]))
    split_point = int(PR_GROUPS_TRAIN * len(valid_subjects))
    _, train_subjects = train_test_split(valid_subjects, test_size=PR_GROUPS_TRAIN, random_state=seed)
    train_keys = [k for k in valid_keys if k[0] in train_subjects]
    test_keys = [k for k in valid_keys if k[0] not in train_subjects]
    logging.info('There are {} train subjects and {} test subjects'.format(len(train_subjects), len(valid_subjects) - len(train_subjects)))
    logging.info('There are {} train keys and {} test keys'.format(len(train_keys), len(test_keys)))


    # Calc drug's representaion by its distribution over IDC codes
    # scan the train keys
    # create a 'profile' per key (distribution over the ICD9 categories)
    # sum all the 'profiles' that connected to each drug to create a histogram of ICD9 codes distribution per drug
    drug2vec = {drug: np.zeros((18,)) for drug in valid_drugs}
    for key in train_keys:
        profile = np.zeros((18,))
        for sn, icd in enumerate(reversed(key2icd_seq[key])):
            if icd in valid_icd9_codes:
                profile[code2category[icd]] += sn
        for drug in valid_key2drugs[key]:
            drug2vec[drug] += profile

    ####################################################################################################################
    # DEBUG drug2vec representaion
    # drug2vec_norm = {k: v / np.linalg.norm(v) for k, v in drug2vec.items()}
    # drug_embeds = np.stack(list(drug2vec_norm.values()))
    # sim_matrix = cosine_similarity(drug_embeds, drug_embeds)
    # topk5 = np.argsort(-sim_matrix, axis=1)[:, 1:5]
    #
    # ind2drug = {i: drug for i, drug in enumerate(drug2vec_norm.keys())}
    # drug2ind = {drug: i for i, drug in enumerate(drug2vec_norm.keys())}
    #
    # from pharmpy.epc import EPCEngine
    # epe = EPCEngine()
    # ls = {i: epe.get_epc(str(i))['name_generic'] for i in valid_drugs}
    # valid_ndc2name = {}
    # for k, v in ls.items():
    #     if v != 'na':
    #         valid_ndc2name[k] = v
    #
    # ndc_with_names= list((valid_ndc2name.keys()))
    # for q in ndc_with_names:
    #    i = np.where(np.array(valid_drugs) == int(q))[0].item()
    #    top_g = topk5[i, :]
    #    valid_g = [valid_drugs[g] for g in top_g if valid_drugs[g] in ndc_with_names]
    #    if len(valid_g) > 3:
    #        print('q:{} top5:{}'.format(valid_ndc2name[q], [valid_ndc2name[valid_drugs[g]] for g in top_g if valid_drugs[g] in ndc_with_names]))
    #
    # acc_at_k = {1: [], 3: [], 5: [], 10: []}
    # ind2max_cat = {i: np.argmax(v) for i, v in enumerate(drug2vec.values())}
    # for i, q in enumerate(valid_drugs):
    #    cat_q = ind2max_cat[i]
    #    for k in acc_at_k.keys():
    #        top_g = topk5[i, :k]
    #        acc_at_k[k].append(sum([ind2max_cat[g] == cat_q for g in top_g]) / float(k))
    # print([np.array(v).mean() for v in acc_at_k.values()])
    ####################################################################################################################

    drug2ind = {drug: i for i, drug in enumerate(drug2vec.keys())}
    drug_embeds = np.stack(list(drug2vec.values())).astype(np.float32)

    x_train, y_train = create_data(train_keys, method='strict')
    x_test, y_test = create_data(test_keys, method='strict')

    ds_train = SetDataset(records=x_train, is_init=True)
    ds_test = SetDataset(records=x_test, is_init=True)

    eval.save_pickle({'ds_train': ds_train,
                      'y_train': y_train,
                      'ds_test': ds_test,
                      'y_test': y_test},
                     '/home/royhir/projects/SetTrees/eval/mimic/outputs/strict/seed={}_strict_dataset.pkl'.format(seed))
