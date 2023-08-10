from transformers import AutoModel, BertModel, HerbertTokenizerFast, BertTokenizer
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from transformers.adapters import PrefixTuningConfig
import transformers.adapters.composition as ac
from torch import DoubleTensor
from typing import List
import numpy as np
import argparse
import torch
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.9, type=float, help="learning rate decay")
    parser.add_argument("--hidden_dim", default=1024, type=int, help="hidden dimension of the layer after HerBert")
    parser.add_argument("--epochs", default=15, type=int, help="number of epochs")
    parser.add_argument("--gru", default=False, type=bool, help="whether or not to include gru layer")
    parser.add_argument("--bidirectional", default=True, type=bool, help="should the model be bidirectional")
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers of GRU (after HerBert)")
    parser.add_argument("--dropout", default=0.5, type=float, help="specfiy dropout coefficient")
    parser.add_argument("--batch_size", default=2, type=int, help="batch size for training")
    parser.add_argument("--seed", default=1234, type=int, help="seed for more deterministic results")
    parser.add_argument("--mode", default="train", choices=["train", "test", "cache_model", "predict", "mar"], help="set the mode")
    parser.add_argument("--cache_name", default="", type=str, help="Name of the file model will be cached to")
    parser.add_argument("--serialized_model", default=False, type=bool, help="whether or not serialized model is to be loaded")
    parser.add_argument("--utterance", default="Przykład do klasyfikacji", type=str, help="utterance to classify during test_prediction mode")
    parser.add_argument("--output_dim", default=3, type=int,  help="number of classes")
    parser.add_argument("--train_herbert", default=True, type=bool, help="should herbert be trained")
    parser.add_argument("--lr_decay_step_size", default=2, type=int,  help="how often decrease lr")
    parser.add_argument("--dataset", nargs="+", default="/data2/nlp_team/data/sentiment_analysis/pl/bank_A-handlabeled/bank_A.csv", help="dataset to use")
    parser.add_argument("--model_path", default="", type=str, help="path to the pretrained model that will be loaded")
    parser.add_argument("--gpu", default=5, type=int,  help="id of the gpu to train on")#nargs="+",
    parser.add_argument("--workers", default=16, type=int,  help="number of parallel workers for data loader (recommended the same size as batch size)")
    parser.add_argument("--lang", default="pl", choices=["pl", "en", "ru"], help="give info about which language should the model handle")
    parser.add_argument("--stop_bert_training", default=7, type=int, help="number of epochs before training of bert is stopped and only head is trained")
    parser.add_argument("--save_format", default="simple", choices=["simple", "extended"], help="choose whether to save more info or just basic label after prediction")
    parser.add_argument("--save_path", type=str, default="", help="choose path to save predictions to")
    parser.add_argument("--pretrain_path", default="", type=str,  help="path to the pretrained herbert weights that will be loaded")
    args = parser.parse_args()
    return args


def make_weights_for_balanced_classes(sampler, nclasses=3):
    count = [0] * nclasses
    for item in sampler:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i]) if count[i] != 0 else N
    weight = [0] * len(sampler)
    for idx, val in enumerate(sampler):
        weight[idx] = weight_per_class[val[1]]
    return weight


def make_sampler(dataset, weighted_sampler=True, num_of_classes=3):
    indices = list(range(len(dataset)))
    if weighted_sampler:
        # For unbalanced dataset we create a weighted sampler
        sampler = []
        for i in indices:
            sampler.append(dataset[i])
        weights = make_weights_for_balanced_classes(sampler, num_of_classes)
        sampler = WeightedRandomSampler(DoubleTensor(weights), len(weights))
    else:
        sampler = SubsetRandomSampler(indices)
    return sampler


def get_filename_no_path_no_extension(path):
    return path.split("/")[-1].split(".")[0]


def get_pkl_filename(model_path, datasets):
    filename = get_filename_no_path_no_extension(model_path)+"--" if model_path else ""
    for path in datasets:
        filename = f"{filename}-{get_filename_no_path_no_extension(path)}"
    filename = f"checkpoint_m/{filename}.pkl"
    return filename


def get_prediction_data(args):
    utterances = []
    with open(args.dataset, "r") as f:
        for line in f:
            sentences = line.replace("?",".") \
                            .replace("!",".") \
                            .replace("...",".") \
                            .replace("..",".") \
                            .split(".")
            for s in sentences:
                if len(s.strip())>1:
                    utterances.append(s.strip())
    return utterances


def get_model(lang, device):
    model = None
    if lang=='pl':
        model = AutoModel.from_pretrained("allegro/herbert-base-cased").to(device)
        print("Loaded herbert")
    elif lang=='en':
        model = BertModel.from_pretrained("bert-base-cased").to(device)
        print("Loaded bert")
    elif lang=='ru':
        model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational").to(device)
    else:
        raise ValueError(f"SentimentDataset __init__ : lang:{lang} not supported")

    assert model != None
    return model


def get_tokenizer(lang):
    tokenizer = None
    if lang=='pl':
        tokenizer = HerbertTokenizerFast.from_pretrained("allegro/herbert-base-cased")
    elif lang=='en':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif lang=='ru':
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
    else:
        raise ValueError(f"SentimentDataset __init__ : lang:{lang} not supported")

    assert tokenizer != None
    return tokenizer


def prep_adapters(model, train_adapter, load_adapter, adapter_root_dir_path, adapter_name_list, prefix_length=30):
    if load_adapter and adapter_name_list:
        print("Loading adapters...")
        for adapter_name in adapter_name_list:
            adapter_dir = os.path.join(adapter_root_dir_path, adapter_name)
            if os.path.exists(adapter_dir):
                adapter_name = model.load_adapter(adapter_dir)
                model.set_active_adapters(adapter_name)

        print(adapter_dir)
        print(adapter_name)
        print(model.get_adapter(adapter_name))
        print("Adapters loaded")
         
    elif train_adapter:
        print("Creating adapter...")
        task_name = "small_dataset_sentiment_finetuning"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            # adapter_config = AdapterConfig.load(
            #     adapter_name,
            #     reduction_factor=1,
            # )
            adapter_config = PrefixTuningConfig(flat=False, prefix_length=prefix_length)
            model.add_adapter(task_name, config=adapter_config)
            lang_adapter_name = None

        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])

        # Set the adapters to be used in every forward pass
        if lang_adapter_name:
            model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        else:
            model.set_active_adapters(task_name)
    

def initialize_model(
    device, 
    args=None,
    dataloader_train=None, 
    dataloader_val=None, 
    dataloader_test=None, 
    model_path=None, 
    predicting=True,
    pretrain_path=None,
    lang="pl",
    train_adapter=False,
    load_adapter=False,
    adapter_root_dir_path: str = "",
    adapter_name_list: List[str] = ["small_dataset_sentiment_prediction"],
):
    if not model_path:
        model_path = args.model_path
    
    if not args:
        model = HerbertSentiment(
            train_dataloader = dataloader_train,
            val_dataloader = dataloader_val,
            test_dataloader = dataloader_test,
            predicting = predicting,
            device = device,
            lang=lang,
            train_adapter=train_adapter,
        ).to(device)
    else:
        model = HerbertSentiment(
            gru = args.gru,
            lang = args.lang,
            dropout = args.dropout,
            hidden_dim = args.hidden_dim,
            bidirectional = args.bidirectional,
            output_dim = args.output_dim,
            n_layers = args.n_layers,
            lr = args.lr,
            lr_decay = args.lr_decay,
            lr_decay_step_size = args.lr_decay_step_size,
            model_path = model_path,
            pretrain_path = args.pretrain_path,
            train_herbert = args.train_herbert,
            dataset = args.dataset,
            train_dataloader = dataloader_train,
            val_dataloader = dataloader_val,
            test_dataloader = dataloader_test,
            predicting = predicting,
            device = device,
            train_adapter=train_adapter,
        ).to(device)
        
    # preparing in case args is none
    lang = args.lang if args else lang
    # initialize bert
    model.herbert = get_model(lang, device)
    # initialize tokenizer
    model.tokenizer = get_tokenizer(lang)
    # load pkl for whole model
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded state dict from {model_path}")
        
    # loading only bert weights pretrained on a different task
    if pretrain_path:
        print(f"Loading bert pretrain weights from {pretrain_path}")
        model.herbert.load_state_dict(torch.load(pretrain_path, map_location=device))
    
    # handling adapters
    if train_adapter or load_adapter:
        prep_adapters(model.herbert,
                      train_adapter=train_adapter, 
                      load_adapter=load_adapter, 
                      adapter_root_dir_path=adapter_root_dir_path, 
                      adapter_name_list=adapter_name_list)
    
    return model


def prep_prediction_output(predictions):
    labels = [get_labels() for _ in range(len(predictions))]
    # sorting the output according to pro/fzarnecki/answer_prediction/checkpoint_mbabilities (descending)
    zipped = [zip(l,p) for l,p in zip(labels, predictions)]
    sorted_ = [sorted(z, reverse=True, key=lambda pair: pair[1]) for z in zipped]
    tuples = [zip(*s) for s in sorted_]
    labels_sorted, probas_sorted = [],[]
    for tuples_ in tuples:
        l, p = [list(tuple) for tuple in tuples_]
        labels_sorted.append(l)
        probas_sorted.append(p)
    return labels_sorted, probas_sorted


def save_prediction_to_json(save_p, utterances, labels_sorted, probas_sorted, save_format):
#     save_p = "".join(dataset.split(".")[:-1]) + "_sentiment"

    if save_format == "extended":
        save_p += ".json"
        res = []
        for i in range(len(utterances)):
            res.append(
                {
                    "utterance": utterances[i].replace("\"", ""),
                    "chosen_label": labels_sorted[i][0],
                    "chosen_label_prob": probas_sorted[i][0],
                    "labels_sorted": labels_sorted[i],
                    "probabilities_sorted": probas_sorted[i], 
                }
            )

        try:
            with open(save_p, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"save_prediction_to_json: unable to save to {save_p}")
            raise e
    else:
        save_p += ".csv"
        try:
            with open(save_p, "w", encoding="utf-8") as f:
                for i in range(len(utterances)):
                    f.write(f"{utterances[i]}, {str(labels_sorted[i][0])}\n")
            return True
        except IOError as e:
            print(f"save_prediction_to_json: unable to save to {save_p}")
            raise e


def get_labels():
    return ["__negatywny__", "__neutralny__", "__pozytywny__"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    

def set_bert_training(con, model):
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = con


def get_dataset_filename(dataset_path):
    filename = dataset_path.split('/')[-1]
    print("Filename: ", filename)
    return filename


from model.Herbert import HerbertSentiment