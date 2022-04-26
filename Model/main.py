import sys
sys.path.append("..")

import numpy as np
import torch
import pickle
import pandas as pd
import torch.nn as nn
from torch.nn import DataParallel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoTokenizer
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse
from sklearn.utils.class_weight import compute_class_weight
import Classifier.train as trainer
from Classifier.predict import predict
import Classifier.global_parameters as gp
from Classifier.model import BERT_Arch
from Classifier.data_creator import *
from Classifier.tokenizer import *


def get_compar_test_set(data, directive=""):
    json_file_name = "/home/reemh/LIGHTBITS/DB_cetus_test/database.json"
    ids = []
    with open(json_file_name, 'r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        for i, key in enumerate(file_data):
            if i % 1000 == 0:
                print("Progress, completed: {0}%".format(i * 100 / len(file_data)))
            if not key == "key":
                if directive == "":
                    ids.append(file_data[key][gp.KEY_ID])
                else:
                    pragma = db_read_string_from_file(file_data[key][gp.KEY_OPENMP])
                    if directive in pragma:
                        ids.append(file_data[key][gp.KEY_ID])
    remove_ids = []
    new_data = {}
    new_data = gp.Data()
    new_data.test_labels = data.test_labels.copy()
    new_data.test = data.test.copy()
    new_data.test_ids = data.test_ids.copy()
    for i in range(len(data.test_ids)):
        found = False
        for j in range(len(ids)):
            # print(i, len(data.test_ids))
            if data.test_ids[i] == ids[j]:
                found = True
                break
        if not found:
            ind = 0
            for j, id in enumerate (new_data.test_ids):
                if id == data.test_ids[i]:
                    ind = j
                    break
            new_data.test.pop(ind)
            new_data.test_labels.pop(ind)
            new_data.test_ids.pop(ind)
            # new_data.test_labels.pop(i)
            # new_data.test_ids.pop(i)

    print("LENGTH TEST:", len(new_data.test_ids))
    return new_data


def reshuffle_data(data_old):

    df = {'text': [], 'label': []}
    for i in range(len(data_old.train_labels)):
        df['text'].append(data_old.train[i])
        df['label'].append(data_old.train_labels[i])

    for i in range(len(data_old.val_labels)):
        df['text'].append(data_old.val[i])
        df['label'].append(data_old.val_labels[i])

    for i in range(len(data_old.test_labels)):
        df['text'].append(data_old.test[i])
        df['label'].append(data_old.test_labels[i])

    data.train, temp_text, data.train_labels, temp_labels = train_test_split(df['text'],
                                                                             df['label'],
                                                                             random_state = 2021,
                                                                             test_size = 0.2,
                                                                             stratify = df['label'])
    # From the temp we extract the val and the test 50% of the 30% which is 15% of data each
    data.val, data.test, data.val_labels, data.test_labels = train_test_split(temp_text,
                                                                              temp_labels,
                                                                              random_state = 2021,
                                                                              test_size = 0.3,
                                                                              stratify = temp_labels)
    if not isinstance(data.train, list):
        data.train = data.train.tolist()
        data.val = data.val.tolist()
        data.test = data.test.tolist()
        data.train_labels = data.train_labels.tolist()
        data.val_labels = data.val_labels.tolist()
        data.test_labels = data.test_labels.tolist()
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=None, type=str,
                        dest='config_file', help='The file of the hyper parameters.')
    parser.add_argument('--train', default=False, action="store_true",
                        dest='train', help='Train phase.')
    parser.add_argument('--predict', default=False, action="store_true",
                        dest='predict', help='Predict phase.')
    parser.add_argument('--save', default="", type = str,
                        dest='save', help='Save tokenize phase.')
    parser.add_argument('--multiple_gpu', default=False, action = "store_true",
                        dest='multiple_gpu', help='Number of gpus')
    parser.add_argument('--out', default="saved_weights.pt", type=str,
                        dest='out', help='Saved model name.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default="/home/reemh/LIGHTBITS/DB/", type=str,
                        dest='data_dir', help='The Directory of the data.')
    parser.add_argument('--data_type', default="", type=str,
                        dest='data_type', help='The type of read.')
    parser.add_argument('--max_len', default=0, type=int,
                        dest='max_len', help='The type of read.')
    parser.add_argument('--specific_directive', default="reduction", type=str,
                        dest='max_len', help='The type of read.')
    parser.add_argument('--reshuffle', dest='reshuffle',action = "store_true", default=False)

    args = parser.parse_args()

    # CREATE CONFIG DIC
    config = {}
    # config["data_type"] = args.data_type
    config["data_dir"] = args.data_dir
    max_len = args.max_len

    # Read pickle, i assume it is already tokenized!
    if args.data_dir.endswith(".pkl"):
        with open(args.data_dir, 'rb') as f:
            data = pickle.load(f)
    else:
        data = pd.read_csv(args.data_dir)
        data.head()
    if args.reshuffle:
        print("@@@@@@@@@@@@@@ Reshuffle!!! @@@@@@@@@@@@@@")
        data = reshuffle_data(data)

    torch.cuda.empty_cache()
    # print(torch)
    device = torch.device("cuda")


    # seq_len = [len(i.split()) for i in data.train_text]
    # pd.Series(seq_len).hist(bins = 30)
    # model = AutoModelForMaskedLM.from_pretrained("cl-nagoya/defsent-roberta-base-cls")
    model_pretained_name = "NTUYG/DeepSCC-RoBERTa" #'bert-base-uncased'
    # model_pretained_name = 'bert-base-uncased'
    pt_model = AutoModel.from_pretrained(model_pretained_name)

    batch_size = 32
    # Convert lists to tensors..
    if args.train:
        print("Example of data: \n", data.train[126])
        train, _ = deepscc_tokenizer(data.train, args.max_len, model_pretained_name)
        val, _ = deepscc_tokenizer(data.val, args.max_len, model_pretained_name)
        train_seq = torch.tensor(train['input_ids'])
        train_mask = torch.tensor(train['attention_mask'])
        train_y = torch.tensor(data.train_labels)
        # wrap tensors
        train_data = TensorDataset(train_seq, train_mask, train_y)
        # sampler for sampling the data during training
        train_sampler = RandomSampler(train_data)
        # dataLoader for train set
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

        val_seq = torch.tensor(val['input_ids'])
        val_mask = torch.tensor(val['attention_mask'])
        val_y = torch.tensor(data.val_labels)
        # wrap tensors
        val_data = TensorDataset(val_seq, val_mask, val_y)
        # sampler for sampling the data during training
        val_sampler = SequentialSampler(val_data)
        # dataLoader for validation set
        val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

    if args.predict:
        print("Example of data:\n", data.test[126])

        # indices = []
        # data_test = []
        # label_test = []
        # for i, dat in enumerate(data.test):
        #     if len(dat.split()) < 51:
        #         indices.append(i)
        # for i in indices:
        #     data_test.append(data.test[i])
        #     label_test.append(data.test_labels[i])
        data_test = data.test
        label_test = data.test_labels
        test, _ = deepscc_tokenizer(data_test, args.max_len, model_pretained_name)
        maxx = len(test['input_ids'])
        test_seq = torch.tensor(test['input_ids'])
        test_mask = torch.tensor(test['attention_mask'])
        test_y = torch.tensor(label_test)
        test_data = TensorDataset(test_seq, test_mask, test_y)
        test_sampler = SequentialSampler(test_seq)
        test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = batch_size)
        test_to_show = {'label': [], 'id': [], 'input_ids': []}
        test_to_show['label'].extend(data.test_labels)
        test_to_show['id'].extend(data.test_ids)
        test_to_show['input_ids'].extend(test['input_ids'])


    # freeze all the parameters - I.E DO NOT UPDATE PRE-TRAINED MODEL FROM NEW TRAINING
    # for param in pt_model.parameters():
        # param.requires_grad = False
        # print (param.requires_grad)

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(pt_model)

    # push the model to GPU
    if args.multiple_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-5)

    # There is a class imbalance in our dataset. The majority of the observations are not spam.
    # So, we will first compute class weights for the labels in the train set
    # and then pass these weights to the loss function so that it takes care of the class imbalance.
    if args.train:
        class_weights = compute_class_weight('balanced', np.unique(data.train_labels), data.train_labels)
        print("Class Weights:", class_weights)
        # converting list of class weights to a tensor
        weights = torch.tensor(class_weights, dtype=torch.float)
        # push to GPU
        weights = weights.to(device)
        # define the loss function
        cross_entropy = nn.NLLLoss(weight=weights)
    # number of training epochs
    epochs = 15
    # print("Summary:")
    # print("Train:", len(train[0]))
    # print("Valid:", len(val[0]))
    if args.train:
        trainer.train(model, epochs, train_dataloader, device, cross_entropy, optimizer, val_dataloader, args.out)
    # for each epoch
    if args.predict:
        predict(model, device, test_dataloader, test_y, args.out, test_to_show)

# load weights of best model
    # path = 'saved_weights.pt'
    # model.load_state_dict(torch.load(path))
    #
    # # get predictions for test data
    # with torch.no_grad():
    #   preds = model(test_seq.to(device), test_mask.to(device))
    #   preds = preds.detach().cpu().numpy()
    #
    # preds = np.argmax(preds, axis = 1)
    # print(classification_report(test_y, preds))
    # function to train the model
