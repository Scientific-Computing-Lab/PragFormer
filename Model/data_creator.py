import sys
sys.path.append("..")

import pickle as pkl
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Classifier.global_parameters as gp

import time
from Classifier.utils import *
import argparse
# from Classifier.tokenizer import *
import ForPragmaExtractor.visitors as visitor
import ForPragmaExtractor.global_parameters as global_parameters
from pycparser import parse_file, c_ast, c_generator

# Create
DATA_CHOICES = ["as_text", "as_normalized", "as_ast", "as_ast_normalized", "as_ast_reduction", "as_ast_private", "as_ast_dynamic", "as_ast_shared"]

VAR_PREFIX = "var"
ARR_PREFIX = "arr"
FUNC_PREFIX = "func"
STRUCT_PREFIX = "struct"
generator = c_generator.CGenerator()
id_v = visitor.CounterIdVisitor()
replacer = visitor.ReplaceIdsVisitor(VAR_PREFIX, ARR_PREFIX, STRUCT_PREFIX, FUNC_PREFIX)
"""
TODO:
Create a main function that will have an abstract workflow:
1) create pickle
2) create database

in create pickle mode, it should parse the database, split the data, tokenize it and save it
in create database it should generally call the database.insert function -- it gets as an input PragmaForTuple, which can 
easily get from the picle file (in db) and project name, which can be parsed from the original DB
Only thing that is different between database creators is how to create the PragmaForTuple.. 
"""

# TODO:
# 1) Remove pragma omp parallel for from all, it is unnecesary
# 2) Check data for "fake omp"
# 3) Check if the word var1 in the pragma and code the same annotation
class DataCreator:
    def __init__(self, clause = ""):
        self.data = gp.Data()
        self.df = {'label': [], 'text': [], 'id': []}
        self.clause = clause
        pass
    
    def get_pragma(self, pragma_text):
        if self.clause == "":
            if pragma_text == "":
                return 0
            else:
                return 1
        else:
            if pragma_text == "":
                print("ERROR, shouldn't reach this if-clause")
                exit(1)
            if self.clause in pragma_text:
                return 1
            else:
                return 0

        
    def parse_database(self, path_to_db):
        print("AS SIMPLE TEXT")
        if self.clause:
            print("WITH CLAUSE", self.clause)
        json_file_name = os.path.join(path_to_db, "database.json")

        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                if not key == "key":
                    pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                    code = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                    if not should_add_pragma(file_data[key], self.clause):
                        continue
                    pragma = self.get_pragma(pragma)
                    self.df['label'].append(pragma)
                    self.df['text'].append(get_code_from_pickle(file_data[key][gp.KEY_PICKLE]))
                    self.df['id'].append(file_data[key]["id"])
        print ("NUMBER OF SET", len(self.df['text']))

    def split_and_tokenize_data(self):

        with open("../data/as_text_25.pkl", 'rb') as f:
            data = pickle.load(f)
        for i, val in enumerate(self.df['id']):
            if val in data.train_ids:
                self.data.train_ids.append(val)
                self.data.train.append(self.df['text'][i])
                self.data.train_labels.append(self.df['label'][i])
            elif val in data.val_ids:
                self.data.val_ids.append(val)
                self.data.val.append(self.df['text'][i])
                self.data.val_labels.append(self.df['label'][i])
            elif val in data.test_ids:
                self.data.test_ids.append(val)
                self.data.test.append(self.df['text'][i])
                self.data.test_labels.append(self.df['label'][i])

        # pack ids
        #
        # data = []
        # for i in range(len(self.df['text'])):
        #     data.append([self.df['text'][i], self.df['id'][i]])
        # data_train, temp_text, self.data.train_labels, temp_labels = train_test_split(data, self.df['label'],
        #                                                                     random_state = 2018,
        #                                                                     test_size = 0.25,
        #                                                                     stratify = self.df['label'])
        # # From the temp we extract the val and the test 50% of the 30% which is 15% of data each
        # data_val, data_test, self.data.val_labels, self.data.test_labels = train_test_split(temp_text, temp_labels,
        #                                                                 random_state = 2018,
        #                                                                 test_size = 0.5,
        #                                                                 stratify = temp_labels)
        # # unpack ids
        # for i, dat in enumerate(data_train):
        #     self.data.train.append(dat[0])
        #     self.data.train_ids.append(dat[1])
        #
        # for i, dat in enumerate(data_val):
        #     self.data.val.append(dat[0])
        #     self.data.val_ids.append(dat[1])
        #
        # for i, dat in enumerate(data_test):
        #     self.data.test.append(dat[0])
        #     self.data.test_ids.append(dat[1])
        #
        # if not isinstance(self.data.train, list):
        #     self.data.train = self.data.train.tolist()
        #     self.data.val = self.data.val.tolist()
        #     self.data.test = self.data.test.tolist()
        #     self.data.train_labels = self.data.train_labels.tolist()
        #     self.data.val_labels = self.data.val_labels.tolist()
        #     self.data.test_labels = self.data.test_labels.tolist()

        print("Examples:")
        print(self.data.train_ids[100], self.data.train_labels[100], self.data.train[100])
        print(self.data.train_ids[567], self.data.train_labels[567], self.data.train[567])
        print(self.data.train_ids[2500], self.data.train_labels[2500], self.data.train[2500])
        print(self.data.train_ids[7000], self.data.train_labels[7000], self.data.train[7000])
        # print(self.data.train_ids[11000], self.data.train_labels[11000], self.data.train[11000])
        # self.data.train, _ = deepscc_tokenizer(self.data.train, 50)  # max_len'
        # self.data.val, _ = deepscc_tokenizer(self.data.val, 50)  # max_len
        # self.data.test, _ = deepscc_tokenizer(self.data.test, 50)  # max_len


class DataCreatorLeClair(DataCreator):
    def __init__(self):
        self.df = {}
        self.data = {}
        pass

    def parse_database(self, path_to_db):
        print("AS LECLAIR")
        json_file_name = os.path.join(path_to_db, "database.json")
        col = ['label', 'text']
        self.df = {'label': [], 'text': []}
        num = 0
        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                if not key == "key":
                    pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                    code = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                    if pragma == "":
                        continue
                    if len(code) > 50:
                        num = num + 1
                    pragma.replace('pragma', '')
                    pragma.replace('omp', '')
                    pragma.replace('parallel', '')
                    pragma.replace('for', '')
                    self.df['label'].append(pragma)
                    self.df['text'].append(code)
        print(num, len(self.df['label']))

    def split_and_tokenize_data(self):



        train_text, temp_text, train_labels, temp_labels = train_test_split(self.df['text'], self.df['label'],
                                                                            random_state = 2018,
                                                                            test_size = 0.3)
        # From the temp we extract the val and the test 50% of the 30% which is 15% of data each
        val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                        random_state = 2018,
                                                                        test_size = 0.5)
        # train_text, dat_vocab_size = deepscc_tokenizer(train_text, 50)  # max_len'
        # train_labels, com_vocab_size = deepscc_tokenizer(train_labels, 50)  # max_len
        # val_text, vocab_size = deepscc_tokenizer(val_text, 50)  # max_len
        # val_labels, vocab_size = deepscc_tokenizer(val_labels, 50)  # max_len
        # test_text, vocab_size = deepscc_tokenizer(test_text, 50)  # max_len
        # test_labels, vocab_size = deepscc_tokenizer(test_labels, 50)  # max_len'

        self.data['ctrain'] = train_labels
        self.data['cval'] = val_labels
        self.data['ctest'] = test_labels
        self.data['dtrain'] = train_text
        self.data['dval'] = val_text
        self.data['dtest'] = test_text
        # self.data['dat_vocab_size'] = dat_vocab_size
        # self.data['com_vocab_size'] = com_vocab_size


class DataCreatorFakePragmaAndNormalize(DataCreator):
    def parse_database(self, path_to_db):
        """
        :param input: input file -- should be the pickle or the json
        :param output: the output file (pickle of dict)
        :return: nothing
        """
        print("AS NORMALIZED TEXT")
        if self.clause:
            print("WITH CLAUSE", self.clause)
        json_file_name = os.path.join(path_to_db, "database.json")
        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)

            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                code   = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                if not should_add_pragma(file_data[key], self.clause):
                    continue
                pragma = self.get_pragma(pragma)
                pickle_file = file_data[key][gp.KEY_PICKLE]
                self.df['text'].append(normalize_code_as_string(pickle_file))
                self.df['label'].append(pragma)
                self.df['id'].append(file_data[key]["id"])


class DataCreatorAST(DataCreator):
    def parse_database(self, path_to_db):
        print("AS SIMPLE AST")
        if self.clause:
            print("WITH CLAUSE", self.clause)
        """
        :param input: input file -- should be the pickle or the json
        :param output: the output file (pickle of dict)
        :return: nothing
        """
        long_pragma = 0
        json_file_name = os.path.join(path_to_db, "database.json")
        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)

            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                code   = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                if not should_add_pragma(file_data[key], self.clause):
                    continue
                pragma = self.get_pragma(pragma)
                pickle_file = file_data[key][gp.KEY_PICKLE]
                self.df['text'].append(code_as_ast(pickle_file))
                self.df['label'].append(pragma)
                self.df['id'].append(file_data[key]["id"])

                num_pragma = num_pragma + 1
                    # exit(1)


class DataCreatorASTNormalized(DataCreator):
    def parse_database(self, path_to_db):
        from transformers import AutoTokenizer
        print("AS NORMALIZED AST")
        if self.clause:
            print("WITH CLAUSE", self.clause)
        """
        :param input: input file -- should be the pickle or the json
        :param output: the output file (pickle of dict)
        :return: nothing
        """
        long_pragma = 0
        json_file_name = os.path.join(path_to_db, "database.json")
        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)

            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                code   = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                if not should_add_pragma(file_data[key], self.clause):
                    continue
                pragma = self.get_pragma(pragma)
                pickle_file = file_data[key][gp.KEY_PICKLE]
                self.df['text'].append(normalize_code_as_ast(pickle_file))
                self.df['label'].append(pragma)
                self.df['id'].append(file_data[key]["id"])

                num_pragma = num_pragma + 1
                    # exit(1)
        print("Number of total directives:", num_pragma)
        print("Examples:")
        print(self.df['label'][100], self.df['text'][100])
        print(self.df['label'][567], self.df['text'][567])
        print(self.df['label'][2500], self.df['text'][2500])
        print(self.df['label'][7000], self.df['text'][7000])
        print(self.df['label'][9000], self.df['text'][9000])


class DataCreatorASTClause(DataCreator):
    def __init__(self, clause):
        self.df = {'label': [], 'text': [], 'id': []}
        self.data = gp.Data()
        self.clause = clause

    def parse_database(self, path_to_db):
        from transformers import AutoTokenizer

        """
        :param input: input file -- should be the pickle or the json
        :param output: the output file (pickle of dict)
        :return: nothing
        """

        json_file_name = os.path.join(path_to_db, "database.json")
        num_pragma = 0
        with open(json_file_name, 'r') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            self.df = {'label': [], 'text': []}

            # Join new_data with file_data inside emp_details
            for i, key in enumerate(file_data):
                if i % 1000 == 0:
                    print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                code   = db_read_string_from_file(file_data[key][gp.KEY_CODE])
                if self.clause in pragma:
                    pragma = 1
                else:
                    pragma = 0
                if should_add_pragma(file_data[key]) or pragma == "": # code is a full line
                    continue
                else:
                    pickle_file = file_data[key][gp.KEY_PICKLE]
                    self.df['text'].append(normalize_code_as_ast(pickle_file))
                    self.df['label'].append(pragma)
                    self.df['id'].append(file_data[key]["id"])

                    # exit(1)
        print("Number of directives:", num_pragma)
        print("Examples:")
        print(self.df['label'][100], self.df['text'][100])
        print(self.df['label'][567], self.df['text'][567])
        print(self.df['label'][2500], self.df['text'][2500])
        print(self.df['label'][7000], self.df['text'][7000])
        print(self.df['label'][11000], self.df['text'][11000])


def should_add_pragma(file_data_key, clause):
    pragma = db_read_string_from_file(file_data_key[gp.KEY_OPENMP])
    code = db_read_string_from_file(file_data_key[gp.KEY_CODE])
    pickle_file = file_data_key[gp.KEY_PICKLE]
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
    max_len_ast = visitor.get_length_ast(for_ast)
    if max_len_ast > should_add_pragma.max_ast:
        should_add_pragma.counter = should_add_pragma.counter + 1
        return False
    if not clause:
        if is_fake_loop(code) and pragma != "":  # code is a full line
            should_add_pragma.counter = should_add_pragma.counter + 1
            return False
        return True
    else: # if we are given a clause as input, we don't add non-openmp directives
        if is_fake_loop(code) or pragma == "":  # code is a full line or doesn't contain the clause
            should_add_pragma.counter = should_add_pragma.counter + 1
            return False
        return True


def normalize_code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        # for_ast.show()
        # print(normalize_code_as_string.generator.visit(for_ast))
        # for_ast.show()
        # counts in an array the name and identifiers of the code
        id_v.reset()
        id_v.visit(for_ast)
        # Replace the names..
        replacer.reset(id_v.ids, id_v.array,
                                             id_v.struct, id_v.func)
        replacer.visit(for_ast)
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces
        # print(ast_no_whitespaces)
        # print(normalize_code_as_string.generator.visit(for_ast))

        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


def code_as_ast(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        with open('temp.txt', 'w') as f:
            for_ast.show(buf=f)
        with open('temp.txt', 'r') as f:
            ast = f.readlines()

        ast_no_whitespaces = [a.strip() for a in ast] # kill all whitespaces and \n
        ast_one_line = " " + " ".join(ast_no_whitespaces)
        return ast_one_line


def normalize_code_as_string(pickle_file):
    # print (pickle_file)
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        # for_ast.show()
        # counts in an array the name and identifiers of the code
        id_v.reset()
        id_v.visit(for_ast)
        # Replace the names..
        replacer.reset(id_v.ids, id_v.array,
                                                id_v.struct, id_v.func)
        replacer.visit(for_ast)
        return generator.visit(for_ast)


def get_code_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
        return generator.visit(for_ast)

def get_function_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        code_data = ""
        for n in pragmafor_tuple.inner_nodes:
            code_data = code_data + "\n" + generator.visit(n)
        return code_data


def initialize_pycparser():
    from pycparser import parse_file, c_ast, c_generator
    should_add_pragma.counter = 0


def is_fake_loop(code_directive):
    code_directive = code_directive.split("\n")
    # print(len(code_directive))
    if len(code_directive) < 4 and (code_directive[1].strip() == '' or code_directive[1].strip() == ';'):
        return True
    return False


def data_creator(config):
    if config['data_type'] not in DATA_CHOICES:
        print("WRONG DATA TYPE")
        print("Choose: ", DATA_CHOICES)
        exit(1)

    should_add_pragma.max_ast = config["max_ast"]
    should_add_pragma.clause = config["clause"]
    path_to_db = config["data_dir"]
    parse_type = config["data_type"]
    save = config["save"]
    initialize_pycparser()
    t0 = time.time()
    print("Creating Data with max:", should_add_pragma.max_ast)
    # we create a dictionairy that has the text and label from the DB
    # as test as normalized as ast as ast_normalized
    if parse_type == DATA_CHOICES[0]:
        creator = DataCreator(config["clause"])
    if parse_type == DATA_CHOICES[1]:
        creator = DataCreatorFakePragmaAndNormalize(config["clause"])
    if parse_type == DATA_CHOICES[2]:
        creator = DataCreatorAST(config["clause"])
    if parse_type == DATA_CHOICES[3]:
        creator = DataCreatorASTNormalized(config["clause"])

    creator.parse_database(path_to_db)
    creator.split_and_tokenize_data()

    new_json = save
    with open(new_json, 'wb') as f:
        pkl.dump(creator.data, f)

    print("Number of Training set:", len(creator.data.train))
    print("Number of Valid set:", len(creator.data.val))
    print("Number of Test set:", len(creator.data.test))
    num_directives1 = [a for a in creator.data.train_labels if a == 1]
    num_directives2 = [a for a in creator.data.val_labels if a == 1]
    num_directives3 = [a for a in creator.data.test_labels if a == 1]
    print("Number of Directives:", len(num_directives1) + len(num_directives2) + len(num_directives3))
    print("Number of Directives Removed:", should_add_pragma.counter)
    print("Elapsed time:", time.time() - t0)


def statistics(config):
    path_to_db = config["data_dir"]
    json_file_name = os.path.join(path_to_db, "database.json")
    DIRECTIVES = ["reduction", "private", "dynamic", "shared", "lastprivate", "firstprivate", "collapse"]
    num_occur = [0] * len(DIRECTIVES)
    total = 0
    max_len_ast = []
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)

        # Join new_data with file_data inside emp_details
        for i, key in enumerate(file_data):
            if i % 1000 == 0:
                print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
            pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
            code = db_read_string_from_file(file_data[key][gp.KEY_CODE])
            pickle_file = file_data[key][gp.KEY_PICKLE]
            with open(pickle_file, 'rb') as f:
                pragmafor_tuple = pkl.load(f)  #
                for_ast = pragmafor_tuple.for_node

            max_len_ast.append(visitor.get_length_ast(for_ast))

    n, bins, patches = plt.hist(x = max_len_ast, bins = 'auto', color = '#0504aa',
                                alpha = 0.7, rwidth = 0.85)
    print("MAX FREQ:", n.max())
    plt.xlabel('Length AST')
    plt.ylabel('Occurences')
    plt.show()

    # pd.Series(seq_len).hist(bins = 30)


def statistics2(config):
    path_to_db = config["data_dir"]
    json_file_name = os.path.join(path_to_db, "database.json")
    num_pragma = 0
    DIRECTIVES = ["reduction", "private", "dynamic", "shared", "lastprivate", "firstprivate", "collapse"]
    num_occur = [0] * len(DIRECTIVES)
    total = 0
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)

        # Join new_data with file_data inside emp_details
        for i, key in enumerate(file_data):
            if i % 1000 == 0:
                print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
            pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
            code = db_read_string_from_file(file_data[key][gp.KEY_CODE])
            if is_fake_loop(code) and pragma != "" or pragma == "":  # code is a full line
                continue

            total = total + 1
            for i, clause in enumerate(DIRECTIVES):
                if clause in pragma:
                    num_occur[i] = num_occur[i] + 1
    print("Total directives: ", total)
    for i, clause in enumerate(DIRECTIVES):
        print("Number of ", clause, " :", num_occur[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default=None, type=str,
                        dest='create_type', help='The file of the hyper parameters.')
    parser.add_argument('--save', default=None, type=str,
                        dest='save', help='Train phase.')
    parser.add_argument('--max_ast', default=25, type=int,
                        dest='max_ast', help='Train phase.')
    parser.add_argument('--statistics', default=False, action = "store_true",
                        dest='statistics', help='Train phase.')
    parser.add_argument('--all', default = False, action = "store_true",
                        dest = 'all', help = 'Train phase.')
    parser.add_argument('--clause', default = "", type = str,
                        dest = 'clause', help = 'specific directive.')
    args = parser.parse_args()
    print(args)
    config = {}
    config['data_type'] = args.create_type
    config["data_dir"] = "/home/reemh/LIGHTBITS/DB/"
    config['save'] = args.save
    config['max_ast'] = args.max_ast
    config["clause"] = args.clause
    if args.statistics:
        print("Statistics of the DB")
        statistics2(config)
    else:
        if args.all:
            CHOICES = ["as_text", "as_normalized", "as_ast", "as_ast_normalized"]
            for c in CHOICES:
                config['data_type'] = c
                if config["clause"]:
                    config['save'] = "../data/" + c + "_" + config["clause"] + "_" + str(args.max_ast) + ".pkl"
                else:
                    config['save'] = "../data/" + c + "_" + str(args.max_ast) + ".pkl"

                data_creator(config)
        else:
            data_creator(config)