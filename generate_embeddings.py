import argparse
import torch
import json
from tqdm import tqdm
from utils.utils import initialize_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="", help="dataset to use")
    parser.add_argument("--model_path", default="", type=str, help="path to the pretrained model that will be loaded")
    parser.add_argument("--gpu", default=5, type=int,  help="id of the gpu to train on")
    parser.add_argument("--save_path", type=str, default="", help="choose path to save predictions to")
    args = parser.parse_args()
    return args


def main(args):
    if not args.model_path and not args.serialized_model:
        raise ValueError("predict: model_path or model not provided")
    if not args.dataset:
        raise ValueError("predict: dataset not provided")
    if not args.save_path:
        raise Exception("predict: no save path provided")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = initialize_model(
        device = device, 
        model_path = args.model_path, 
        predicting = True
    )

    # read data
    utterances = []
    with open(args.dataset, "r") as f:
        for line in f:
            utterances.append(line.replace("\n", "").strip())
    print("Loaded prediction data of size:", len(utterances))

    # generate embeddings
    embeddings = []
    with torch.no_grad():
        for utt in tqdm(utterances):
            emb = model.embed([utt]).cpu().detach().numpy().tolist()
            embeddings.append(emb[0])
    assert len(utterances) == len(embeddings)

    # save results
    save_dicts_list = []
    for text,emb in zip(utterances, embeddings):
        save_dicts_list.append(
            {
                "utterance": text,
                "embedding": emb
            }
        )
    with open(args.save_path, "w") as f:
        json.dump(save_dicts_list, f)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)