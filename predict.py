from utils.utils import (
    parse_args,
    set_seed, 
    initialize_model, 
    prep_prediction_output,
    save_prediction_to_json,
)
from tqdm import tqdm
import joblib
import torch
import os


def predict(args):
    if not args.model_path and not args.serialized_model:
        raise ValueError("predict: model_path or model not provided")
    if not args.dataset:
        raise ValueError("predict: dataset not provided")
    if not args.save_path:
        raise Exception("predict: no save path provided")
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.serialized_model:   # fully ready model being loaded
        model = joblib.load(args.model_path)
        print(f"Loaded {args.model_path} model.")
    else:   # initializing model with cached weights
        model = initialize_model(
            args = args,
            dataloader_train = None, 
            dataloader_val = None, 
            dataloader_test = None, 
            device = device, 
            model_path = args.model_path, 
            predicting = True,
        )

    # utterances = ["Bardzo ciekawe zagadnienie",
    #              "Gdybyśmy mogli pomóc to byśmy to zrobili, ale niestety nic nie poradzimy, że ukradli panu kartę",
    #              "Super, bardzo się cieszę!",
    #              "Nasze badania są wyjątkowe pod wieloma względami, szczególnie elementy, które obejmują",
    #              "Raczej nie jest to nic specjalnego, zwłaszcza rozważając te nieudane próby",
    #              "Obraz hiszpańskiego malarza Francisca Goi, znajdujący się w prywatnej kolekcji barona Edmonda Adolphe de Rothschilda w Szwajcarii."]
    utterances = []
    with open(args.dataset[0], "r") as f:
        for line in f:
            utterances.append(line)

    print("Loaded prediction data of size:", len(utterances))

    predictions = []
    with torch.no_grad():
        for utt in tqdm(utterances):
            res = model([utt]).cpu().detach().numpy().tolist()
            predictions.append(res[0])

    labels_sorted, probs_sorted = prep_prediction_output(predictions)

    assert len(utterances) == len(predictions) == len(labels_sorted) == len(probs_sorted)

    saved = save_prediction_to_json(args.save_path, utterances, labels_sorted, probs_sorted, args.save_format)

    return saved


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(42)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    _ = predict(args)

if __name__ == "__main__":
    main()
    exit(0)
