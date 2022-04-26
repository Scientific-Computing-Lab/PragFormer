from __future__ import print_function

import json
import sys
import re
import os
import fileinput

sys.path.extend(['.', '..'])
from global_parameters import *

from pycparser import parse_file, c_ast, c_generator
from pycparser.plyparser import Coord
from visitors import *
import pickle
import glob


def prep_file(file):
    found = False
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("//"):
                continue
            if "#include" in line and "omp.h" in line:
                lines[i] = "#include \"omp.h\"\n"
                found = True
                break
    if not found:
        return
    print ("EDITING FILE")
    with open(file, "w") as f:
        f.writelines(lines)


def get_files_from_repo(folder_path):
    header_files = glob.glob(folder_path + "/**/*.h", recursive = True)
    header_folders = []
    max_folders = 100
    for header in header_files:
        header_dir = os.path.dirname(header)
        if header_dir not in header_folders and len(header_folders) < max_folders:
            header_folders.append(header_dir)
    return glob.glob(folder_path + "/**/*.c", recursive = True), header_folders


def check_openmp_pragma(file: os.path):
    num_pragmas = 0
    try:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith("//"):
                    continue
                if "#pragma" in line and "omp" in line and "parallel" in line and "for" in line:
                    num_pragmas = num_pragmas + 1
                if "#pragma" in line and "omp" in line and ("critical" in line or "atomic" in line):
                    num_pragmas = num_pragmas - 1
        return num_pragmas
    except:
        print("FAILED TO PARSE IN check_openmp_pragma")
        f = open(BAD_FILE, "a+")
        f.writelines("@files_handler.py:" + file + "\n")
        f.close()
        return False


def check_folder_exists_in_log(log_path:str, folder: str):
    folder = folder.strip()
    with open(log_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if os.path.samefile(line, folder):
                return True
    return False