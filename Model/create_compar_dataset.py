import sys
sys.path.append("..")

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
from Classifier.data_creator import should_add_pragma, get_code_from_pickle, get_function_from_pickle
from ForPragmaExtractor.global_parameters import PragmaForTuple, FAKE_HEADER_PATH
import subprocess as sub_process
import shutil

# Create
from ForPragmaExtractor.database import Database
generator = c_generator.CGenerator()
id_v = visitor.CounterIdVisitor()
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


def get_dataset_test(path_to_pickle, path_to_db, path_to_new_db, override, p4a):
    if p4a == False:
        with open(path_to_pickle, 'rb') as f:
            dataset = pkl.load(f)
            test_data = dataset.test
            valid_data = dataset.val
            test_label = dataset.test_labels
            valid_label = dataset.val_labels
            valid_ids = dataset.val_ids
            test_ids = dataset.test_ids
    else:
        test_ids = []

    json_file_name = os.path.join(path_to_db, "database.json")
    db = Database(os.path.join(os.path.abspath(path_to_new_db), "database")
                  , os.path.join(os.path.abspath(path_to_new_db), "database.json"), override)
    failed = 0
    sucess = 0
    error = 0
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        for i, key in enumerate(file_data):
            if file_data[key]["id"] in test_ids or test_ids == []:
                if db.check_if_repo_exists(key):
                    continue
                pickle_file = file_data[key][gp.KEY_PICKLE]
                code_as_string = prep_code_as_string(file_data, key)
                
                pragma = "!"
                prefix = "*"
                for kk in range(2):
                    code_as_string = prep_code_as_string(file_data, key, prefix)
                    if p4a:
                        pragma = exeute_autoparallel_p4a(code_as_string)
                    else:
                        pragma = exeute_autoparallel(code_as_string)
                    prefix = prefix + "*"
                    if pragma != "!":
                        print("Fixed problem!")
                        break
                if pragma == "!":
                    print("Compiler failed")
                    error = error + 1
                    continue
                if pragma == "":
                    pragma = c_ast.Pragma(pragma)
                    pragma.string = ""
                    failed = failed + 1
                    print("Failed")
                else:
                    sucess = sucess + 1
                    print("Success!")
                with open(pickle_file, 'rb') as f:
                    pragmafor_tuple = pkl.load(f)  #
                    for_ast = pragmafor_tuple.for_node
                    inner_nodes = pragmafor_tuple.inner_nodes
                pragma_for_tup = PragmaForTuple(pragma, for_ast)
                pragma_for_tup.set_inner_nodes(inner_nodes)
                db.insert(pragma_for_tup, key, file_data[key]["id"])
    print("Errors:", error, " out of ", failed + sucess)


def create_db(path_to_db, path_to_new_db, override):
    json_file_name = os.path.join(path_to_db, "database.json")
    db = Database(os.path.join(os.path.abspath(path_to_new_db), "database")
                  , os.path.join(os.path.abspath(path_to_new_db), "database.json"), override)
    failed = 0
    sucess = 0
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        for i, key in enumerate(file_data):
            if i % 50 == 0:
                print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
                print("Failed:", failed)
                print("Success:", sucess)
            if not key == "key":
                pickle_file = file_data[key][gp.KEY_PICKLE]

                code_as_string = get_code_from_pickle(pickle_file)
                pragma = exeute_autoparallel(code_as_string)
                if pragma == "":
                    pragma = c_ast.Pragma(pragma)
                    pragma.string = ""
                    failed = failed + 1
                else:
                    sucess = sucess + 1
                with open(pickle_file, 'rb') as f:
                    pragmafor_tuple = pkl.load(f)  #
                    for_ast = pragmafor_tuple.for_node
                    inner_nodes = pragmafor_tuple.inner_nodes
                pragma_for_tup = PragmaForTuple(pragma, for_ast)
                pragma_for_tup.set_inner_nodes(inner_nodes)
                db.insert(pragma_for_tup, key, file_data[key]["id"])


def prep_code_as_string(file_data, key, prefix="*"):
    pickle_file = file_data[key][gp.KEY_PICKLE]
    # print("WORKING ON", key)
    code_as_string = get_code_from_pickle(pickle_file)
    id_visitor = visitor.CounterIdVisitor()
    id_visitor.reset()
    with open(pickle_file, 'rb') as f:
        pragmafor_tuple = pkl.load(f)  #
        for_ast = pragmafor_tuple.for_node
    id_visitor.visit(for_ast)
    # code_as_string = "int main() {\n" + code_as_string + "\n}\n"
    real_ids = visitor.ReplaceIdsVisitor("a", "a", "a", "a")
    real_ids.reset(id_visitor.ids, id_visitor.array, id_visitor.struct, id_visitor.func)

    decl = "int"
    decl = construct_variable_decl(real_ids.var, decl)
    decl = decl + "int"
    decl = construct_variable_decl(real_ids.array, decl, prefix=prefix)
    decl = decl + "int"
    decl = construct_variable_decl(real_ids.struct, decl, prefix=prefix)

    code_as_string = "int main() {\n" + decl + "\n" + code_as_string + "\n}\n"
    code_as_string = code_as_string + "\n" + get_function_from_pickle(pickle_file)
    return code_as_string


def construct_variable_decl(ids, code, prefix=""):
    if len(ids) == 0:
        # print("EMPTY CODE")
        return code + ";\n"
    for id in ids:
        if isinstance(id, str):
            code = code + " " + prefix + id + ", "

    code = code[0:-2] + ";\n"
    return code


def exeute_autoparallel(code_as_string):
    # returns the pragma
    if os.path.isdir('cetus_output'):
        shutil.rmtree('cetus_output')

    with open("tmp.c", "w") as f:
        # f.writelines("int main() {\n")
        f.writelines(code_as_string + "\n")
        # f.writelines("}\n")
        # print(code_as_string)
    process = sub_process.Popen(['cetus', '-alias=3', '-preprocessor=cpp -C -I/home/reemh/CLPP/fake_headers ', 'tmp.c'], stdout = sub_process.PIPE, stderr = sub_process.PIPE)
    out, err = process.communicate()
    process.wait()
    if not os.path.isdir('cetus_output'):
        # print(err)
        return "!"

    # if err != "":
    #     print("Error:", err)
    #     return ""
    # print(out)
    file = os.path.join("cetus_output/", "tmp.c")
    cpp_args = ['-nostdinc', '-E', r'-I' + FAKE_HEADER_PATH]
    ast = parse_file(file, use_cpp = True, cpp_path = 'mpicc', cpp_args = cpp_args)
    # Get openmp pragma..
    pragma_for_visit = visitor.PragmaForVisitor()
    pragma_for_visit.visit(ast)
    pragmas = pragma_for_visit.pragmas

    if len(pragmas) != 1: # no pragma
        return ""
    else:
        print("Compiler Success!")
        return pragmas[0]


def exeute_autoparallel_p4a(code_as_string):
    # returns the pragma
    os.system('rm -rf P4A* && rm tmp.p4a.c')

    with open("tmp.c", "w") as f:
        # f.writelines("int main() {\n")
        f.writelines(code_as_string + "\n")
        # f.writelines("}\n")
        # print(code_as_string)
    process = sub_process.Popen(['p4a', '-O', '-I/home/reemh/CLPP/fake_headers ', '--no-pointer-aliasing', 'tmp.c']
                                , stdout = sub_process.PIPE, stderr = sub_process.PIPE)#, shell=True)

    out, err = process.communicate()
    process.wait()
    if not os.path.isfile('tmp.p4a.c'):
        print("Compiler failed...")
        print(out, err)
        exit(1)
        return "!"

    # if err != "":
    #     print("Error:", err)
    #     return ""
    # print(out)
    file = os.path.join('tmp.p4a.c')
    cpp_args = ['-nostdinc', '-E', r'-I' + FAKE_HEADER_PATH]
    ast = parse_file(file, use_cpp = True, cpp_path = 'mpicc', cpp_args = cpp_args)
    # Get openmp pragma..
    pragma_for_visit = visitor.PragmaForVisitor()
    pragma_for_visit.visit(ast)
    pragmas = pragma_for_visit.pragmas

    if len(pragmas) != 1: # no pragma
        return ""
    else:
        print("Compiler Success!")
        return pragmas[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default=None, type=str,
                        dest='create_type', help='The file of the hyper parameters.')
    parser.add_argument('--pickle', default=None, type=str,
                        dest='pickle', help='Train phase.')
    parser.add_argument('--statistics', default=False, action = "store_true",
                        dest='statistics', help='Train phase.')
    parser.add_argument('--override', default=False, action = "store_true",
                        dest='override', help='Train phase.')
    parser.add_argument('--test', default=False, action = "store_true",
                        dest='test', help='Train phase.')
    parser.add_argument('--p4a', default=False, action = "store_true",
                        dest='p4a', help='Train phase.')

    args = parser.parse_args()
    print(args)
    config = {}
    config['data_type'] = args.create_type
    config["data_dir"] = "/home/reemh/LIGHTBITS/DB/"
    if args.test:
        if args.p4a:
            config['new_data_dir'] = "/home/reemh/LIGHTBITS/DB_p4a_test/"
            config["data_dir"] = "/home/reemh/LIGHTBITS/DB_cetus_test/"

        else:
            config['new_data_dir'] = "/home/reemh/LIGHTBITS/DB_compar_private/"

        get_dataset_test(args.pickle, config["data_dir"], config['new_data_dir'], args.override, args.p4a)
    else:
        config['new_data_dir'] = "/home/reemh/LIGHTBITS/DB_COMPAR/"

        create_db(config["data_dir"], config['new_data_dir'], args.override)