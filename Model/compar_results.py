import sys
from collections import OrderedDict

sys.path.append("..")
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import Classifier.utils as utils
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from Classifier.utils import *
import ForPragmaExtractor.global_parameters as gp
import json
import pickle as pkl
import json
import os

import time
from Classifier.utils import *
import Classifier.global_parameters as gp
import argparse
# from Classifier.tokenizer import *
import ForPragmaExtractor.visitors as visitor
from pycparser import parse_file, c_ast, c_generator
from Classifier.data_creator import should_add_pragma, get_code_from_pickle
from ForPragmaExtractor.global_parameters import PragmaForTuple, FAKE_HEADER_PATH
import subprocess as sub_process
import shutil
# Create
from ForPragmaExtractor.database import Database



def check_has_function(file_data_key):
    pickle_path = file_data_key[gp.KEY_PICKLE]
    with open(pickle_path, 'rb') as f:
        data = pkl.load(f)
    if data.inner_nodes == []:
        return False
    else:
        return True



with open("../data/as_text_25.pkl", 'rb') as f:
    dataset = pkl.load(f)
    test_data = dataset.test
    valid_data = dataset.val
    test_label = dataset.test_labels
    valid_label = dataset.val_labels
    valid_ids = dataset.val_ids
    test_ids = dataset.test_ids

json_file_name = "/home/reemh/LIGHTBITS/DB_cetus_test/database.json"
json_file_real = "/home/reemh/LIGHTBITS/DB/database.json"
y_compar = {'label': [], 'id': []}
y_compar_private = {'label': [], 'id': []}
y_compar_reduction = {'label': [], 'id': []}
y_real = {'label': [], 'id': []}
y_real_private = {'label': [], 'id': []}
y_real_reduction = {'label': [], 'id': []}
with open(json_file_name, 'r') as file:
    # First we load existing data into a dict.
    file_data = json.load(file)
    file_data_compar = file_data.copy()
    # Join new_data with file_data inside emp_details
    # for i, key in enumerate(file_data):
    #     if i % 1000 == 0:
    #         print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
    #     if not key == "key":
    #         pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
    #         code = db_read_string_from_file(file_data[key][gp.KEY_CODE])
    #         if pragma == "":
    #             y_compar['label'].append(0)
    #         else:
    #             # if "reduction" in pragma:
    #             #     y_compar_reduction['label'].append(1)
    #             #     y_compar_reduction['id'].append(file_data[key]['id'])
    #             # else:
    #             #     y_compar_reduction['label'].append(0)
    #             #     y_compar_reduction['id'].append(file_data[key]['id'])
    #             #
    #             # if "private" in pragma:
    #             #     y_compar_private['label'].append(1)
    #             #     y_compar_private['id'].append(file_data[key]['id'])
    #             # else:
    #             #     y_compar_private['label'].append(0)
    #             #     y_compar_private['id'].append(file_data[key]['id'])
    #
    #             y_compar['label'].append(1)
    #
    #         y_compar['id'].append(file_data[key]['id'])
redutction = 0
with open(json_file_real, 'r') as file:
    # First we load existing data into a dict.
    file_data = json.load(file)
    function_side_effect = 0
    # Join new_data with file_data inside emp_details
    index_last_add = 0
    for i, key in enumerate(file_data):
        if not file_data[key]['id'] in test_ids:
            continue
        if not key == "key":
            pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
            try:
                compar_pragma = db_read_string_from_file(file_data_compar[key][gp.KEY_OPENMP])
            except:
                # IF you want to comapre compar without setting 0 to all the examples, here is where u should skip to the next iteration
                compar_pragma = ""

            if compar_pragma == "":
                y_compar['label'].append(0)
            else:
                y_compar['label'].append(1)
            y_compar['id'].append(file_data[key]['id'])
            if pragma == "":
                y_real['label'].append(0)
            else:
                y_real['label'].append(1)
                if "reduction" in pragma:
                    y_real_reduction['label'].append(1)
                    redutction = redutction + 1
                else:
                    y_real_reduction['label'].append(0)
                if "private" in pragma:
                    y_real_private['label'].append(1)
                else:
                    y_real_private['label'].append(0)
                # Add compar result
                if "reduction" in compar_pragma:
                    y_compar_reduction['label'].append(1)
                else:
                    y_compar_reduction['label'].append(0)
                if "private" in compar_pragma:
                    y_compar_private['label'].append(1)
                else:
                    y_compar_private['label'].append(0)
            y_real['id'].append(file_data[key]['id'])



y_test = []
y_pred = []
for i, val1 in enumerate(y_real['id']):
    for j, val2 in enumerate(y_compar['id']):
        if val1 == val2:
            y_test.append(y_real['label'][i])
            y_pred.append(y_compar['label'][j])
            if i != j:
                print("Error, not comparing the same ids")
                exit(1)


print("Number of test Compar did :", len(y_compar['label']))
print("Number of test Compar missed :", 3547 - len(y_compar['label']))
cls_rpt = classification_report(y_pred, y_test)
print(cls_rpt)

print("Report of Private:")
y_test = y_compar_private['label']
y_pred = y_real_private['label']
cls_rpt = classification_report(y_pred, y_test)
print(cls_rpt)

print("Report of Reduction:")
y_test = y_compar_reduction['label']
y_pred = y_real_reduction['label']
cls_rpt = classification_report(y_pred, y_test)
print(cls_rpt)




