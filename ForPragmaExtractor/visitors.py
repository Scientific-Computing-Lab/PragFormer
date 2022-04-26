from __future__ import print_function
import json
import sys
import re
sys.path.extend(['.', '..'])

from pycparser import parse_file, c_ast, c_generator
from pycparser.plyparser import Coord
from collections import OrderedDict


def get_length_ast(node):
    if not len(node.children()):
        return 1
    lengther = []
    for c in node:
        lengther.append(get_length_ast(c) + 1)
    return max(lengther)


# Puts in array all the ids found, function name(calls), array and structs
class ReplaceIdsVisitor(c_ast.NodeVisitor):
    def __init__(self, var_prefix, array_prefix, struct_prefix, func_prefix):
        self.var_prefix = var_prefix
        self.array_prefix = array_prefix
        self.struct_prefix = struct_prefix
        self.func_prefix = func_prefix

    def visit_ID(self, node):
        for i, val in enumerate(self.array):
            if node.name == val:
                node.name = self.array_prefix + str(i)
                return
        for i, val in enumerate(self.struct):
            if node.name == val:
                node.name = self.struct_prefix + str(i)
                return
        for i, val in enumerate(self.func):
            if node.name == val:
                node.name = self.func_prefix + str(i)
                return
        for i, val in enumerate(self.var):
            if node.name == val:
                node.name = self.var_prefix + str(i)
                return
        print("Error id")
        exit(1)

    def visit_Decl(self, node):
        for i, val in enumerate(self.array):
            if node.name == val:
                node.name = self.array_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.struct):
            if node.name == val:
                node.name = self.struct_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.func):
            if node.name == val:
                node.name = self.func_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.var):
            if node.name == val:
                node.name = self.var_prefix + str(i)
                self.generic_visit(node)
        # print("Error decl")
        # exit(1)

    def visit_TypeDecl(self, node):
        # print (node)
        if not node.declname:
            self.generic_visit(node)
        for i, val in enumerate(self.array):
            if node.declname == val:
                node.declname = self.array_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.struct):
            if node.declname == val:
                node.declname = self.struct_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.func):
            if node.declname == val:
                node.declname = self.func_prefix + str(i)
                self.generic_visit(node)
        for i, val in enumerate(self.var):
            if node.declname == val:
                node.declname = self.var_prefix + str(i)
                self.generic_visit(node)
        # print("Error type decl")
        #exit(1)

    # def visit_ArrayDecl(self, node):
    #     print(node)
    #     for i, val in enumerate(self.array):
    #         if node.type.declname == val:
    #             node.type.declname = self.array_prefix + str(i)
    #             self.generic_visit(node)
    #     for i, val in enumerate(self.struct):
    #         if node.type.declname == val:
    #             node.type.declname = self.struct_prefix + str(i)
    #             self.generic_visit(node)
    #     for i, val in enumerate(self.func):
    #         if node.type.declname == val:
    #             node.type.declname = self.func_prefix + str(i)
    #             self.generic_visit(node)
    #     for i, val in enumerate(self.var):
    #         if node.type.declname == val:
    #             node.type.declname = self.var_prefix + str(i)
    #             self.generic_visit(node)
    #     # print("Error array decl")
    #     # exit(1)

    def reset(self, var, array, struct, func):
        # remove duplicates..
        self.var = list(OrderedDict.fromkeys(var))
        self.array = list(OrderedDict.fromkeys(array))
        self.struct = list(OrderedDict.fromkeys(struct))
        self.func = list(OrderedDict.fromkeys(func))

        # now remove from self.var all the names from array struct and func
        self.var = [v for v in self.var if v not in self.array]
        self.var = [v for v in self.var if v not in self.struct]
        self.var = [v for v in self.var if v not in self.func]


# Puts in array all the ids found, function name(calls), array and structs
class CounterIdVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.reset()

    def visit_ID(self, node):
        # print("ID:", node.name)
        # if node.name != "i" and node.name != "j":
        if node.name:
            self.ids.append(node.name)

    def visit_FuncCall(self, node):
        # print("FuncCall:", node)
        if isinstance(node.name, c_ast.UnaryOp):
            if node.name.op == '*':
                self.generic_visit(node)
        else:
            self.func.append(node.name.name)
            self.generic_visit(node)

    def visit_ArrayRef(self, node):
        if isinstance(node.name, c_ast.BinaryOp):
            self.generic_visit(node)
            return
        # CASTING
        if isinstance(node.name, c_ast.Cast):
            if isinstance(node.name.expr, c_ast.ID) or isinstance(node.name.expr, c_ast.ArrayRef):
                name = node.name.expr.name
            if isinstance(node.name.expr, c_ast.StructRef):
                name = node.name.expr.field
            if isinstance(node.name.expr, c_ast.UnaryOp):
                if isinstance(node.name.expr.expr, c_ast.ArrayRef):
                    self.generic_visit(node)
                    return
                name = node.name.expr.expr.name
        # ARRAY OF STRUCT
        if isinstance(node.name, c_ast.StructRef):
            name = node.name.field
        # NORMAL
        if isinstance(node.name, c_ast.ID):
            name = node.name.name
        # UNARY OP WHICH IS BASICALLY CAST TO STRUCT..
        if isinstance(node.name, c_ast.UnaryOp):
            if isinstance(node.name.expr, c_ast.StructRef):
                name = node.name.expr.field
            if isinstance(node.name.expr, c_ast.ID):
                name = node.name.expr.name
        if isinstance(node.name, c_ast.ArrayRef):
            # if it is an array of arrays (2d array etc, we will just continue to the next expr..)
            self.generic_visit(node)
            return
        # print (node.name)
        try:
            if isinstance(name, c_ast.ID):
                name = name.name
            self.array.append(name)
            self.generic_visit(node)
        except:
            print(node.name)
            exit(1)

    def visit_ArrayDecl(self, node):
        if isinstance(node.type, c_ast.PtrDecl):
            name = node.type.type.declname
        if isinstance(node.type, c_ast.TypeDecl):
            name = node.type.declname
        if isinstance(node.type, c_ast.ArrayDecl):
            # if it is an array of arrays (2d array etc, we will just continue to the next expr..)
            self.generic_visit(node)
            return
        # print (node.name)
        try:
            self.array.append(name)
            self.generic_visit(node)
        except:
            print(node)
            exit(1)

    def visit_StructRef(self, node):
        if isinstance(node.name, c_ast.ID):
            name = node.name.name
            self.struct.append(name)
        self.generic_visit(node)

    def reset(self):
        self.ids = []
        self.func = []
        self.array = []
        self.struct = []


class LengthVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.max_len = 0
        self.curr_len =0

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        self.curr_len = self.curr_len + 1
        for c in node:
            self.visit(c)


class ForVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.out_for_loop_found = True
        self.generator = c_generator.CGenerator()

    def visit_For(self, node):
        self.nodes.append(node)
        #  ("FOUND A FOR")
        # print("############################### start  ####################################")
        # print("TYPE:", type(node))
        # print ("WE VISIT A NODE:", self.generator.visit(node))
        # printprint ("############################### DONE  ####################################")

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        self.found = False
        for c in node:
            self.visit(c)


# Travels the node of a outer for-loop that has an openmp directive, and it if finds an atomic or critical inside - returns true!
class ForLoopChecker(c_ast.NodeVisitor):
    """
    Class that travels the node of a for loop that has an openmp directive to find an atomic
    """
    def __init__(self):
        self.found = False

    def reset(self):
        self.found = False

    def visit_Pragma(self, node):
        if "omp" in node.string and ("atomic" in node.string or "barri" in node.string or "critical" in node.string):
            self.found = True
        else:
            self.generic_visit(node)


# Travels the whole code and tries to find a pragma omp parallel for that afterwards there is a for loop
class PragmaForVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.pragmas = []
        self.found = False

    def visit_For(self, node):
        if self.found:
            self.nodes.append(node)
            self.found = False
        else:
            self.generic_visit(node)

    def visit_Pragma(self, node):
        if "parallel" in node.string and "for" in node.string and "omp" in node.string:
            self.pragmas.append(node)
            self.found = True

    def generic_visit(self, node):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        self.found = False
        for c in node:
            self.visit(c)


# appends all function defenitions
class FuncDefVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.func_def = []
        self.generator = c_generator.CGenerator()

    def visit_FuncDef(self, node):
        if isinstance(node, c_ast.FuncDef):
            self.func_def.append(node)


# Class to find and visit all the function calls inside a For loop that should contain the OpenMP pragama above it
class Visitor(c_ast.NodeVisitor):
    def __init__(self):
        self.nodes = []
        self.func_calls = []
        self.c_gen = c_generator.CGenerator()

    # def visit_Pragma(self, node):
    #     print ("Pragma inside a for loop, can't handle, exiting")
    #     exit(1)

    def visit_FuncCall(self, node):
        self.func_calls.append(node)
        self.generic_visit(node)

    def visit(self, node):
        """ Visit a node.
        """

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor
        # print (type(node))
        return visitor(node)

