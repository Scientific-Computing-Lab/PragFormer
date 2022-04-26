import os
import json
import pickle


def db_read_string_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            return "".join(f.readlines())
    except:
        return ""


def get_pragma_clause(pragma, clause):
    num_clause = pragma.split(' ').count(clause + '(')
    if num_clause > 1:
        print("Two directives")
        exit(1)
    pos = find_word(pragma, clause)
    if pos == -1:
        return ""
    pos_start, pos_end = get_parenthese_scope(pragma, start=pos)
    return (clause + pragma[pos_start:pos_end+1])


def find_word(line, word, start=0):
    for i in range(start, len(line)):
        if line[i:i + len(word)] == word:
            return i
    return -1

# Returns the position of the start and end of the paratheses
def get_parenthese_scope(line, start=0):
    stack = []
    in_sograim = ""
    start_parenthese = 0
    end_parenthese = 0
    for i in range(start, len(line)):
        letter = line[i]
        if letter == "(":  # good, we found the start of sograim
            if len(stack) == 0: # No sograim was found yet, this is the first one...
                start_parenthese = i
            stack.append("(")

            continue
        elif letter == ")":
            stack.pop()
            if len(stack) == 0:  # Empty
                end_parenthese = i
                return start_parenthese, end_parenthese
        elif len(stack) != 0:  # Stack has to be filled with the relevant word for us to add it to the sograim...
            in_sograim += letter
    return start_parenthese, end_parenthese

def read_pickle_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        return data