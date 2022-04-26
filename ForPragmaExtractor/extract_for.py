from __future__ import print_function
import json
import sys
import re
sys.path.extend(['.', '..'])
from pycparser.plyparser import Coord
from ForPragmaExtractor.visitors import *
import pickle
import os
from ForPragmaExtractor.global_parameters import PragmaForTuple, FAKE_HEADER_PATH, WITH_NO_PRAGMA, GOOD_FILE, BAD_FILE, DEBUG


class FileParser:
    def __init__(self):
        self.pragma_removed = 0

    def parse(self, file: os.path, headers: list):
        self.pragma_removed = 0
        print ("parsing file:", file)
        # cpp_args = ['-E', r'-Iutils/fake_libc_include']
        cpp_args = ['-nostdinc',  '-E', r'-I' + FAKE_HEADER_PATH]

        for header_file in headers:
            cpp_args.append(r'-I' + header_file)
        try:
            ast = parse_file(file, use_cpp=True, cpp_path='mpicc', cpp_args = cpp_args)
            f = open(GOOD_FILE, "a+")
            f.writelines(file + "\n")
            f.close()
        except Exception as e:
            print ("FAILED TO PARSE!!!!!!")
            print (e)
            # print("Command:", cpp_args)

            f = open(BAD_FILE, "a+")
            f.writelines("@extract_for.py:" + file + "\n")
            f.close()
            return None, None


        # General description of the algorithm:
        # 1. Get all for loops that contain a pragma with "for" and "parallel" in them.
        # 2. Remove from that list all for loops with a pragma of a barriar or atomic - this is usually a bad openmp case..
        # 3. Get all the for-loops without openmp directives (this is done by searching all for loops that do not contain a pragma with "for" or "parallel" in them
        # 4. Find all the function calls inside the two for-loop list (with openmp and without openmp)
        # 5. return a list of PragmaFor tuples that contain the for node and the pragma node (if without openmp then it is none)
        generator = c_generator.CGenerator()
        # for n in relevant_nodes:
        # print(generator.visit(ast))
        # We get all function defenitions for later use
        fdv = FuncDefVisitor()
        fdv.visit(ast)
        func_list = fdv.func_def

        # We first prep the for and pragma tuples then we get the for loops without openmp

        ################################# FOR LOOPS WITH OPENMP: #################################

        # Exctract all loops with a pragma above them
        pfv = PragmaForVisitor() # Should hold all the for loops nodes  that contain a pragma above them
        pfv.visit(ast) # Get the most outer for loops
        for_loops_with_openmp = pfv.nodes
        openmp_directives = pfv.pragmas
        # We now found all nodes that contain a pragma and afterwards a for loop
        # But what happends when it is a for loop with an atomic or barriar inside of it?
        # This is usually a bad directive, so we want to remove them from the for_loops_with_openmp list
        verify_loops = ForLoopChecker()
        remove_for = []
        remove_pragma = []
        for i in range(len(for_loops_with_openmp)):
            verify_loops.reset()
            verify_loops.visit(for_loops_with_openmp[i])
            if verify_loops.found: # we found barriar or atomic inside the loop
                remove_for.append(for_loops_with_openmp[i])
                remove_pragma.append(openmp_directives[i])
                self.pragma_removed = self.pragma_removed + 1
                print ("Removed")
        for_loops_with_openmp = [ele for ele in for_loops_with_openmp if ele not in remove_for]
        openmp_directives = [ele for ele in openmp_directives if ele not in remove_pragma]
        ####################################################### END OF FOR LOOPS WITH OPENMP #############33

        for_nodes_without_openmp = []

        if WITH_NO_PRAGMA:
            pfv = PragmaForVisitor()
            ############################ FOR LOOPS WITHOUT OPENMP ############################
            # Now we extract all the outer for loops
            for_v = ForVisitor()
            for_v.visit(ast)
            all_for_nodes = for_v.nodes
            # Now we should create a list of for loops without
            for for_loop in all_for_nodes:
                # We want to create a list that contains for loops nodes without OpenMP directives.
                # Therefore, we first check if the for loop is in the list of the for loops we gathered above.
                # Of course, the first if is ONLY for the outermost for loop, so we still need to ask the inner for loops...
                # This is handled in the else case
                # if the for loop node is in the list of for loops with pragma in them...
                if for_loop in for_loops_with_openmp:
                    continue
                else:
                    pfv.found = False
                    pfv.visit(for_loop)
                    # The visit of this specific Visitor, is first a pragma, if the following node after the pragma is a for loop
                    # it adds the nodes to some list. In our case, we are searching for the exact opposite case:
                    # given a "father" node of a for loop, if it contains a pragma (for one..) do not add it to the list.
                    # So basically, we can use this function to determine if the some for_loop contains a pragma in it.
                    # Of course this is only true given the above if fails.
                    if pfv.pragmas != []:
                        continue
                # print ("Found for loop without openmp", generator.visit(for_loop))
                for_nodes_without_openmp.append(for_loop)
            if len(all_for_nodes) != len(for_nodes_without_openmp) + len(for_loops_with_openmp):
                print ("You Missed a for loop..")
                f = open(BAD_FILE, "a+")
                f.writelines(file + "\n")
                f.close()
                # return None

        # We now create a list of PragmaForTuple that contains three parameters, the most outer for loop, the relevant pragma
        # and later, the nodes inside the for loop

        pragma_for_tuple = [PragmaForTuple] * (len(for_loops_with_openmp) + len(for_nodes_without_openmp))
        for i in range(len(for_loops_with_openmp) + len(for_nodes_without_openmp)):
            if i < len(for_loops_with_openmp):
                pragma_for_tuple[i] = PragmaForTuple(openmp_directives[i], for_loops_with_openmp[i])
            elif WITH_NO_PRAGMA:
                # add to the pragma for tuple
                j = i - len(for_loops_with_openmp)
                pragma_for_tuple[i] = PragmaForTuple(c_ast.Pragma(""), for_nodes_without_openmp[j])

        # Extract the pragma for loops
        for i, n in enumerate(for_loops_with_openmp):
            node = self.extract_nodes_inside_for_loop(n, func_list)
            pragma_for_tuple[i].set_inner_nodes(node) # yes intentional
        num_pragmas = len(for_loops_with_openmp)
        if WITH_NO_PRAGMA:
            for j, n in enumerate(for_nodes_without_openmp):
                i = j + len(for_loops_with_openmp)
                node = self.extract_nodes_inside_for_loop(n, func_list)
                pragma_for_tuple[i].set_inner_nodes(node)  # yes intentional
        return pragma_for_tuple, num_pragmas

    def extract_nodes_inside_for_loop(self, for_loop: c_ast.For, function_list: list):
        """
        :param for_loop: A For node, should be under Pragma omp parallel for
        :param function_list:  the function list of the whole project
        :return:
        We create a node list that we will serialize afterwards. Each element in the array will be a node that is connected to
        the For Loop.
        For now, the relevant parts are the for loop itself and the function calls!
        """
        generator = c_generator.CGenerator()

        relevant_nodes = []
        v = Visitor()
        v.visit(for_loop)
        func_calls = v.func_calls

        for func_call in func_calls:

            # If we have a decleration of this function, we will add it to the relevant node list
            try:
                relevant_nodes.append([func_def for func_def in function_list if func_call.name.name == func_def.decl.name][0])
            except:
                pass
                # print(generator.visit(func_call))
                # if isinstance(func_call.name, c_ast.FuncDef):
                #     print("Function:", func_call.name.name, "Doesn't exist in the project")
                # else:
                #     print("Not a function but is in something that we thought is a Function call..")
                #     print(generator.visit(func_call))

        # for n in relevant_nodes:
        #     print(generator.visit(n))
        return relevant_nodes

    def generate_c_from_node(self, node):
        generator = c_generator.CGenerator()
        c_code = generator.visit(node)
        return c_code

    def extract_by_regular_parse(self, file):
        openmp_contents = []
        number_of_pragmas = 0

        with open(file, "r") as f:
            lines = f.readlines()
            found_pragma = False
            for i, line in enumerate(lines):
                if found_pragma:
                    openmp_contents[number_of_pragmas][j] = line



                # What if they continue it in the nxt line?
                if "#pragma" in line and "omp" in line and "parallel" in line and "for" in line:
                    found_pragma = True
                    openmp_contents.append([])



