import sys
sys.path.append("..")

import argparse
import os
import json
import pickle as pkl
import Classifier.global_parameters as gp
from Classifier.utils import *
from pycparser import parse_file, c_ast, c_generator
from ForPragmaExtractor.global_parameters import *
import ForPragmaExtractor.visitors as visitor
import ForPragmaExtractor.database as database
import shutil


def create_db_by_directive(path_db, new_path_db, directive_clause):

    if not os.path.isdir(new_path_db):
        os.mkdir(new_path_db)
    json_file_name = os.path.join(path_db, "database.json")
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        for i, key in enumerate(file_data):
            if i % 1000 == 0:
                print("Progress, completed: {0}%".format(i*100 / len(file_data)))
            pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
            if pragma == "":
                continue
            directive = get_pragma_clause(pragma, directive_clause)
            if directive != "":
                # found directive, we will copy the result..
                shutil.copy(src=, dst=)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None, type=str,
                        dest='input', help='The file of the hyper parameters.')
    parser.add_argument('--output', default=None, type=str,
                        dest='output', help='Train phase.')
    parser.add_argument('--type', default = "gal", type = str,
                        dest = 'type', help = 'Train phase.')

    args = parser.parse_args()
    create_db_by_directive(args.input, args.output, "reduction")
