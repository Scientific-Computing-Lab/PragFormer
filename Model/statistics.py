import sys
sys.path.append("..")
import argparse
import pickle
import numpy as np
import Classifier.global_parameters as gp
import time
from Classifier.utils import *
#from Classifier.tokenizer import *
# from transformers import AutoModel
import matplotlib.pyplot as plt

# Create
DATA_CHOICES = ["as_text", "as_normalized", "as_leclair", "as_normalized_no_fake", "as_ast", "as_ast_reduction", "as_ast_private", "as_ast_dynamic", "as_ast_shared"]
DATA_PICKLES = ["../data/as_text_25.pkl", "../data/as_normalized_25.pkl", "../data/as_ast_25.pkl", "../data/as_ast_normalized_25.pkl"]


def get_plot_data(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()
        train_loss = []
        valid_loss = []
        accuracy = []
        epochs = []
        for i, line in enumerate(lines):
            line = line.split()
            train_loss.append(float(line[0]))
            valid_loss.append(float(line[1]))
            accuracy.append(float(line[2]))
            epochs.append(i + 1) # + 1 because it starts with 0
    return epochs, accuracy, train_loss, valid_loss


def plot_train(data_path):
    epochs, accuracy, train_loss, valid_loss = get_plot_data(data_path)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('# Epoch')
    ax1.set_ylabel('%')  # we already handled the x-label with ax1
    lns1 = ax1.plot(epochs, accuracy, label = 'Accuracy', color="red")
    # ax1.tick_params(axis = 'y', labelcolor = color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Loss')
    lns2 = ax2.plot(epochs, train_loss, label = 'Train Loss')
    lns3 = ax2.plot(epochs, valid_loss, label = 'Valid Loss')

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc = 0)
    fig.tight_layout()  # otherw-ise the right y-label is slightly clipped

    plt.show()


def plot_all(datas):
    plt.style.use('/home/reemh/plot_style.txt')
    LABELS = ["Text", "Replaced Text", "AST", "Replaced AST"]
    epochs = []
    accuracy = []
    train_loss = []
    valid_loss = []
    names = []
    for i in range(len(datas)):
        names.append(LABELS[i])
        e, a, tl, vl = get_plot_data(datas[i])
        epochs.append(e)
        accuracy.append(a)
        train_loss.append(tl)
        valid_loss.append(vl)
    clr = ['rs-', 'gx-', 'b^-', 'mo-']

    plot_loss(epochs, train_loss, names, clr, "Train Loss")
    plot_loss(epochs, valid_loss, names, clr, "Valid Loss")
    plot_loss(epochs, accuracy, names, clr, "Accuracy")


def plot_loss(epochs, train_loss, labels, clr, type):
    for i in range(len(epochs)):
        plt.plot(epochs[i], train_loss[i], clr[i], label=labels[i])
    plt.xlabel("# Epochs")
    plt.ylabel(type)
    plt.axvline(x = 9, color='k',linestyle="--", linewidth=1)
    plt.legend()
    plt.xlim([0, 15])
    # plt.xlim([0, 15])
    plt.show()


def create_data_tokenizer(data: gp.Data):
    model_pretained_name = "NTUYG/DeepSCC-RoBERTa"  # 'bert-base-uncased'
    pt_model = AutoModel.from_pretrained(model_pretained_name)
    train, _ = deepscc_tokenizer(data.train, 150, model_pretained_name) #maxlen
    val, _ = deepscc_tokenizer(data.val, 150, model_pretained_name) #maxlen
    test, _ = deepscc_tokenizer(data.test, 150, model_pretained_name)

    new_data = {}
    dat = np.asarray(train["input_ids"])
    dat = dat.flatten(order = 'C')
    new_data["train"] = dat

    dat = np.asarray(val["input_ids"])
    dat = dat.flatten(order = 'C')
    new_data["valid"] = dat
    dat = np.asarray(test["input_ids"])
    dat = dat.flatten(order = 'C')
    new_data["test"] = dat

    total_data = []
    total_data = np.asarray(train["input_ids"]) * np.asarray(train["attention_mask"])
    total_len = len(total_data)
    total_uniqe = 0
    uniq_tokens = 0
    for i, line in enumerate(total_data):
        uniq_tokens = uniq_tokens + len(np.unique(line)) - 1 # for the 0...
    print("NUMBER OF TOKENS IN TRAIN:", uniq_tokens / len(total_data))
    total_uniqe = total_uniqe + uniq_tokens

    uniq_tokens = 0
    total_data = np.asarray(val["input_ids"]) * np.asarray(val["attention_mask"])
    total_len = total_len + len(total_data)
    for i, line in enumerate(total_data):
        uniq_tokens = uniq_tokens + len(np.unique(line))
    print("NUMBER OF TOKENS IN VALID:", uniq_tokens / len(total_data))
    total_uniqe = total_uniqe + uniq_tokens

    total_data = np.asarray(test["input_ids"]) * np.asarray(test["attention_mask"])
    total_len = total_len + len(total_data)
    uniq_tokens = 0
    for i, line in enumerate(total_data):
        uniq_tokens = uniq_tokens + len(np.unique(line))
    print("NUMBER OF TOKENS IN TEST:", uniq_tokens / len(total_data))
    total_uniqe = total_uniqe + uniq_tokens

    return new_data, total_uniqe / total_len


def total_tokens(tokenized_data, avg):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Number Tokens @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    num_tokens = get_total_tokens(tokenized_data)
    for i in range(len(DATA_PICKLES)):
        data = tokenized_data[i]
        print("Case of:", DATA_PICKLES[i])
        print("Average tokens per line:", avg[i])
        # Unique tokens in train valid test
        for key in data.keys():
            print("Number of tokens in {0}: {1}".format(key, len(np.unique(data[key]))))

        np.average(np.unique(data[key]))
        print("Total number of tokens:", num_tokens[i])

def get_total_tokens(tokenized_data):
    tokens = []
    for i in range(len(DATA_PICKLES)):
        data = tokenized_data[i]
        tokens.append(0)
        # Unique tokens in train valid test
        lst = []
        for key in data.keys():
            lst.extend(data[key])
        tokens[i] = len(np.unique(lst))
    return tokens


def histogram_tokens(tokenized_data):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Unique Tokens @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    num_tokens = get_total_tokens(tokenized_data)

    for i in range(len(DATA_PICKLES)):
        data = tokenized_data[i]
        print("\n")
        print("Case of:", DATA_PICKLES[i])

        # Total tokens:
        for key in data.keys():
            print("Number of tokens in {0}: {1}".format(key, len(np.unique(data[key]))))
        print("Total number of tokens:", num_tokens[i])
        # keys are train valid test
        max_val = 0
        unique_unique = data["test"]
        for key in data.keys():
            # print("Histogram of tokens in {0}".format(key))

            n, bins, patches = plt.hist(x = data[key], bins = 'auto', label = key,
                                alpha = 0.7)
            if max_val < n.max():
                max_val = n.max()
            # Calculate
            if key != "test":
                unique_unique = list(set(data["test"]) - set(data[key]))
                print("Found ", len(unique_unique), "token that were in test and not in", key)
            if key == "train":
                data_tot = [None] * (len(data["valid"]) + len(data["test"]))
                for i in range(len(data["valid"])):
                    data_tot[i] = data["valid"][i]
                for i in range(len(data["test"])):
                    data_tot[i + len(data["valid"])] = data["test"][i]
                unique_unique = list(set(data_tot) - set(data[key]))
                print("Out-of-Vocabolary: ", len(unique_unique))

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Done @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


def histogram_of_num_tokens(tokenized_data):
    uniq_tokens = []
    for i in range(len(DATA_PICKLES)):
        data = tokenized_data[i]
        print("Case of:", DATA_PICKLES[i])

        # Unique tokens in train valid test
        for key in data.keys():
            uniq_tokens.append(np.unique(data[key]))


def histograms():
    datas = []
    avg = []
    print("Reading Cute Pickle...")
    for i, pkl in enumerate(DATA_PICKLES):
        with open(pkl, 'rb') as f:
            datas.append(pickle.load(f))

    # Tokenize it and put it in a nice dict
    print("Tokenizing...")
    tokenized_data = []
    for i, pkl in enumerate(DATA_PICKLES):
        a, av = create_data_tokenizer(datas[i])
        avg.append(av)
        tokenized_data.append(a)
    print("Performing data analysis...")
    # print the number of tokens
    total_tokens(tokenized_data, avg)
    histogram_tokens(tokenized_data)


def statistics(config):
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
    parser.add_argument('--data_dir', default=None, type=str,
                        dest='data_dir', help='The file of the hyper parameters.')
    parser.add_argument('--plot', default=False, action="store_true",
                        dest='plot', help='The file of the hyper parameters.')
    args = parser.parse_args()
    all_data = ["as_text.txt", "as_normalize.txt", "as_ast.txt", "as_ast_normalized.txt"]
    # all_data = ["as_text_100_25.txt", "as_normalized_100_25.txt", "as_ast_100_25.txt", "as_ast_normalized_100_25.txt"]
    # all_data = ["as_text_50_25.txt", "as_normalized_50_25.txt", "as_ast_50_25.txt", "as_ast_normalized_50_25.txt"]
    # all_data = ["as_text_bert_25.txt", "as_normalized_bert_25.txt", "as_ast_bert_25.txt", "as_ast_normalized_bert_25.txt"]
    if args.plot:
        plot_all(all_data)
        # plot_train(args.data_dir)
    else:
        histograms()
