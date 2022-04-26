from __future__ import print_function

import json
import sys
import re
import os

sys.path.extend(['.', '..'])
sys.setrecursionlimit(10000)
from pycparser import parse_file, c_ast, c_generator
from pycparser.plyparser import Coord
from visitors import *
import pickle
import database
import argparse
from files_handler import *
import extract_for
import global_parameters as gp


def get_project_name(folder_path: os.path, file_name: os.path, index):
    indir = os.listdir(folder_path)
    # if len(indir) > 1:
    #     print ("Two projects in one username", indir, file_name)
        # exit(1)
    full_project_name = os.path.basename(folder_path) + "_" + indir[0] + "_" + file_name + "_" + str(index)
    return full_project_name


def create_files(override):
    num_pragma = 0
    if override:
        if os.path.exists(gp.PRAGMA_FILES):
            os.remove(gp.PRAGMA_FILES)
        else:
            try:
                f = open(gp.PRAGMA_FILES, "r")
                num_pragma = f.readlines()
                if not num_pragma:
                    num_pragma = 0
                else:
                    num_pragma = int(num_pragma)
                f.close()
            except:
                print("Hmm")
        if os.path.exists(gp.GOOD_FILE):
            os.remove(gp.GOOD_FILE)
        if os.path.exists(gp.BAD_FILE):
            os.remove(gp.BAD_FILE)
        if os.path.exists(gp.LOG_FILE):
            os.remove(gp.LOG_FILE)
    if not os.path.exists(gp.LOG_FILE):
        f = open(gp.LOG_FILE, "w")
        f.close()
    return num_pragma


def main(repos_path, override):
    parser = extract_for.FileParser()
    db = database.Database(gp.DB_PATH, gp.JSON_PATH, override = override)
    num_pragma = create_files(override) # The log, the parsed files, the failed files
    repos = os.listdir(repos_path)


    for i, folder in enumerate(repos):
        folder = os.path.join(repos_path, folder)
        if i % 50 == 0:
            print("################################## FINISHED ANOTHER 50 REPOS #####################################")
            print("PROGRESS:", i*100 / len(repos),"%")
        # LOGGER
        if check_folder_exists_in_log(gp.LOG_FILE, folder): # Exists, skip this folder
            print(folder, ": exists. skipping")
            continue
        print("Working on:", folder)
        files_of_project, header_files = get_files_from_repo(folder)
        inserted_file = True
        # Notes for this loop:
        # 1) Unfortently the code first builds the AST and extracts all the directives and only then it checks if it exists in the DB
        for file in files_of_project:
            file_name = os.path.basename(file)

            ################### LOGIC OF "DO WE WANT TO PARSE THIS FILE?" ##################
            found = False
            for f in gp.EXCLUDE_FOLDERS:
                if file.__contains__(f):
                    found = True
                    break
            if file in gp.EXCLUDE_FILES or found:
                continue
            number_pragmas_parser = check_openmp_pragma(file)
            if number_pragmas_parser <= 0:
                print("REEM CHECKER: No pragma found in file: ", file)
                continue
            project_name = get_project_name(folder, file_name, 0)
            if db.check_if_repo_exists(project_name):
                print("File:", file_name, "exists in the project.")
                continue
            ################### LOGIC OF "DO WE WANT TO PARSE THIS FILE?" ##################

            ######### C PYTHON PARSER ################
            prep_file(file)
            list_of_pragmafor, num_pragmas_cparser = parser.parse(file, header_files)  # List of PragmaForTuple[0:n]
            if not list_of_pragmafor:
                print("C_AST: No pragma found in file")

                continue
            ######### C PYTHON PARSER ################

            if number_pragmas_parser != num_pragmas_cparser and (num_pragmas_cparser + parser.pragma_removed) * 3 < number_pragmas_parser:
                if file not in NESTED_FILES:
                    print(file, ": incosistency with number of pragmas parsed", num_pragmas_cparser, "vs:", number_pragmas_parser)
                    # exit(1)
                    inserted_file = False
                    continue
            print("Adding: ", num_pragmas_cparser, " directives", "out of:", number_pragmas_parser)
            num_pragma = num_pragma + num_pragmas_cparser

            # print ("Found: ", len(list_of_pragmafor), " OpenMP directives")
            for i, pragmafor in enumerate(list_of_pragmafor):
                if i > 100:
                    if not pragmafor.has_openmp():
                        continue
                project_name = get_project_name(folder, file_name, i)
                if db.check_if_repo_exists(project_name):
                    continue
                # It was found that it parses the .h file and their .c as well, so we need to verify that the loop is from
                # the same file.. this can be found in coord!
                coord = pragmafor.get_coord()

                rel_coord = "".join(coord.split(":")[0])
                if file != rel_coord: # Not a pragma and not in file
                    print("NOT REAL FILE")
                    continue
                f = open(gp.PRAGMA_FILES, "w")
                f.writelines(str(num_pragma))
                f.close()
                db.insert(pragmafor, project_name)
        # We finished working on a folder so we added it to the complete list..
        if inserted_file:
            f = open(gp.LOG_FILE, "a+")
            f.writelines(folder + "\n")
            f.close()



# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Runs tests with varying input sizes.')
    parser.add_argument('-repos',
                        dest = 'repos',
                        default = "/home/reemh/CLPP/github-clone-all/repos_final",
                        help = 'Path to the directory containing the runs.')
    parser.add_argument('-db',
                        dest = 'db',
                        default = "/home/reemh/LIGHTBITS/DB/",
                        help = 'Path to the database of files.')
    parser.add_argument('-override',
                        action = 'store_true',
                        dest = 'override',
                        default = False,
                        help = 'Path to the output photos.')
    parser.add_argument('-debug',
                        action='store_true', dest='debug', default = False)

    
    args = parser.parse_args()
    if args.override:
        print("About to remove the current database, are you sure? [y/n]")
        y = input()
        if y == 'y':
            args.override = True
        else:
            args.override = False

    #    if args.debug == True
    DEBUG = args.debug
    gp.DB_PATH = os.path.join(os.path.abspath(args.db), "database")
    gp.JSON_PATH = os.path.join(os.path.abspath(args.db), "database.json")
    gp.LOG_FILE = os.path.join(os.path.abspath(args.db), "log.txt")
    main(args.repos, args.override)
