from torch.utils.data import DataLoader
from dataset.SentimentDataset import SentimentDataset
from utils.utils import (
    parse_args,
    set_seed, 
    initialize_model,
)
import pytorch_lightning as pl
import torch
import os


def test(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    dataset_test = SentimentDataset(dataset_path=args.dataset, lang=args.lang)
    dataloader_test = DataLoader(dataset_test, shuffle = False, batch_size=args.batch_size, num_workers=args.workers)
    
    test_model = initialize_model(
        args=args, 
        dataloader_train=None, 
        dataloader_val=None, 
        dataloader_test=dataloader_test, 
        device=device, 
        model_path=args.model_path, 
        predicting=True,
    )
    
    trainer = pl.Trainer(gpus=[args.gpu])
    trainer.test(test_model, test_dataloaders=dataloader_test)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(42)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    test(args)

if __name__ == "__main__":
    main()
    exit(0)
