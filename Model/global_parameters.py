
import sys
sys.path.append("..")
from ForPragmaExtractor.global_parameters import *
KEY_OPENMP = "pragma"
KEY_PICKLE = "code_pickle"
KEY_CODE = "code"
KEY_ID = 'id'
CUT_LARGER_THAN = 50


class Data:
    def __init__(self):
        self.train = []#train_text
        self.train_labels = []#train_labels
        self.val = []#val_text
        self.val_labels = []#val_labels
        self.test = []# test_text
        self.test_labels = []#test_labels
        self.train_ids = []
        self.val_ids = []
        self.test_ids = []
# class PragmaForTuple:
#     """
#     Class that holds 3 parameters.
#     1) The Pragma c_ast node
#     2) The For c_ast node that corresponds with the pragma
#     3) Inner c_ast nodes that are inside "2" - function calls, variables, etc...
#     """
#     def __init__(self, pragma_node: c_ast.Pragma, node: c_ast.For):
#         self.pragma = pragma_node
#         self.for_node = node
#         self.inner_nodes = []
#
#     def set_inner_nodes(self, inner: list):
#         self.inner_nodes = inner
#
#     def has_openmp(self):
#         coord = "%s" % self.pragma.coord
#         if coord == "None":
#             return False
#         return True
#
#     def get_string_data(self):
#         generator = c_generator.CGenerator()
#         code_data = generator.visit(self.for_node)
#         for n in self.inner_nodes:
#             code_data = code_data + "\n" + generator.visit(n)
#
#         return self.pragma.string, code_data
#
#     def get_coord(self):
#         coord = "%s" % self.pragma.coord
#         if coord == "None":
#             coord = "%s" % self.for_node.coord
#         return coord