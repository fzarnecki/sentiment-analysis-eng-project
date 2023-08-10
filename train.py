from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, random_split
from dataset.SentimentDataset import SentimentDataset
from utils.utils import (
    parse_args,
    set_seed, 
    initialize_model,
    get_pkl_filename,
    make_sampler,
    set_bert_training,
)
import pytorch_lightning as pl
import torch
import os


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    dataset = SentimentDataset(dataset_path=args.dataset, lang=args.lang)

    train_len = round(len(dataset)*0.8)
    val_len = round((len(dataset) - train_len)/2)
    test_len = len(dataset) - train_len - val_len
    train, val, test = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(args.seed))

    train_sampler = make_sampler(train, weighted_sampler=True, num_of_classes=dataset.get_num_classes())
    
    dataloader_train = DataLoader(train, shuffle = False, batch_size=args.batch_size, sampler=train_sampler)
    dataloader_val = DataLoader(val, shuffle = False,batch_size=args.batch_size)
    dataloader_test = DataLoader(test, shuffle = False, batch_size=args.batch_size)

    model = initialize_model(
        args=args,
        dataloader_train=dataloader_train, 
        dataloader_val=dataloader_val, 
        dataloader_test=dataloader_test, 
        device=device, 
        model_path=None, 
        predicting=False,
        pretrain_path=args.pretrain_path,
        train_adapter=False,
        load_adapter=False,
        adapter_root_dir_path="",
        adapter_name_list=["small_dataset_sentiment_prediction"],
    )
    model.herbert.train()
    set_bert_training(True, model.herbert)

    lr_logger = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        callbacks=[lr_logger],
        gpus=[0],
        max_epochs=args.epochs, 
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(model, dataloader_train, dataloader_val)
 
    test_model = initialize_model(
        args=args,
        dataloader_train=None, 
        dataloader_val=None, 
        dataloader_test=dataloader_test, 
        device=device, 
        model_path=get_pkl_filename(args.model_path, args.dataset), 
        predicting=False,
        train_adapter=False,
        load_adapter=False,
        adapter_root_dir_path=get_pkl_filename(args.model_path, args.dataset).split(".")[0] + "_adapters",
        adapter_name_list=["small_dataset_sentiment_finetuning"],
    )
    model.herbert.eval()
    trainer.test(test_model, test_dataloaders=dataloader_test)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(42)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    train(args)

if __name__ == "__main__":
    main()
    exit(0)
