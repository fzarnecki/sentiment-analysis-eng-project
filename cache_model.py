from utils.utils import (
    parse_args,
    set_seed, 
    initialize_model,
)
import joblib
import torch
import os



def cache_model(args):
    if not args.cache_name:
        raise ValueError("cache_model: cache_name not provided")

    device = torch.device('cuda'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = initialize_model(
        args=args,
        dataloader_train=None, 
        dataloader_val=None, 
        dataloader_test=None, 
        device=device, 
        model_path=args.model_path, 
        predicting=True
    )

    joblib.dump(model, 'serialized_models/' + args.cache_name + '.mdl')
    print("Serialized model.")


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(42)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    cache_model(args)

if __name__ == "__main__":
    main()
    exit(0)
