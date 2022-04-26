import sys
sys.path.append("..")
import os
import json
import shutil
import ForPragmaExtractor.global_parameters as gp
import pickle

class Database:
    def __init__(self, path, json_path,  override=False):
        if override:
            inp = 'y'
            if inp == 'y':
                if os.path.isdir(path):
                    shutil.rmtree(path)
                if os.path.isfile(json_path):
                    os.remove(json_path)
                print ("Removing", path)
            else:
                print("You chose no, continue as if not inserted override...")

        if not os.path.isdir(path):
            os.mkdir(path)
        self.db_path = path
        if not os.path.isfile(json_path):
            with open(json_path, 'w') as f:
                dict1 = {"key": 1}
                print (dict1)
                json.dump(dict1, f, indent = 4)
        self.json_path = json_path
        self.override = override

    def insert(self, pragmafor, project_name, id=0):
#def insert(self, pragmafor: gp.PragmaForTuple, project_name: str, id=0):


        folder_path = os.path.join(self.db_path, project_name)
        file_data_pragma, file_data_for_loops = pragmafor.get_string_data()
        original_file = pragmafor.get_coord()
        if self.check_if_repo_exists(project_name):
            print("Folder:", folder_path, " exists")
            return

        # As a part of the database, we create a folder of the project
        os.mkdir(folder_path)

        # Now we copy the pragma.c and code.c that contains the relevant code segments.
        no_openmp = os.path.join(folder_path, gp.FULL_CODE_NAME + ".c")
        pickle_file = os.path.join(folder_path, gp.PICKLE_CODE_NAME + ".pkl")
        with_openmp = os.path.join(folder_path, gp.OPENMP_CODE_NAME + ".c")

        # CREATE AND WRITE THE FILES
        if file_data_for_loops != "":
            f = open(no_openmp, "w")
            f.writelines(file_data_for_loops)
            f.close()
        else:
            print("NO FOR LOOP DATA")
            input()
            no_openmp = ""
        if file_data_pragma != "":
            f = open(with_openmp, "w")
            f.writelines(file_data_pragma)
            f.close()
        else:
            with_openmp = ""
        pickle.dump(pragmafor, open(pickle_file, "wb"), protocol = 2)
        # Now we add project_name to the json (should be the username along with the project name)
        # And add to the that key, the path to pragma.c and code.c
        self._write_json(project_name, gp.FULL_CODE_NAME, no_openmp)
        self._write_json(project_name, gp.OPENMP_CODE_NAME, with_openmp)
        self._write_json(project_name, gp.PICKLE_CODE_NAME, pickle_file)
        self._write_json(project_name, "original", original_file)
        self._write_json(project_name, "id", id)

    def _write_json(self, proj_name, key, new_data):
        filename = self.json_path
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            if proj_name in file_data:
                file_data[proj_name][key] = new_data
            else:
                file_data[proj_name] = {}
                file_data[proj_name][key] = new_data
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)

    def _exists_in_json(self, key1, key2=""):
        filename = self.json_path
        with open(filename, 'r+') as file:
            # First we load existing data into a dict.
            try:
                file_data = json.load(file)
                if key1 in file_data:
                    if not key2:
                        return True
                    return key2 in file_data[key1]
                else:
                    return False
            except:
                return False

    def check_if_repo_exists(self, project_name):
        return self._exists_in_json(project_name) and \
               self._exists_in_json(project_name, gp.FULL_CODE_NAME) and \
               self._exists_in_json(project_name, gp.OPENMP_CODE_NAME)

        # Join new_data with file_data inside emp_details
        # Sets file's current position at offset.
        # convert back to json.

