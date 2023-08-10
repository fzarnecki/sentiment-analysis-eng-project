#from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
#from transformers import XLMTokenizer, RobertaModel, get_constant_schedule_with_warmup
from utils.utils import set_bert_training, get_model, get_tokenizer, get_pkl_filename
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics as tm
import torch.nn as nn
import numpy as np
import torch
import os


class HerbertSentiment(pl.LightningModule):
    
    """Polish Herbert for sentiment analysis"""
    def __init__(
        self,
        gru=False,
        lang="pl",
        dropout=0.5,
        hidden_dim=1024,
        bidirectional=True,
        output_dim=3,
        n_layers=2,
        lr=1e-5,
        lr_decay_step_size=2,
        lr_decay=0.9,
        model_path=None,
        pretrain_path=None,
        train_herbert=True,
        dataset=None,
        train_dataloader=None,
        val_dataloader=None, 
        test_dataloader=None,
        device="cuda",
        predicting=True, #TODO try to remove, maybe use predict mode
        train_adapter=False,
    ):
        super().__init__()

        self.gru = gru
        # initialized in utils
#         self.herbert = get_model(lang, device, pretrain_path)
#         self.tokenizer = get_tokenizer(lang)
#         self.herbert.train()
        # adapters handled in utils
        self.herbert_output_size = 768
        self.softmax = nn.Softmax(1)
        self.dropout = nn.Dropout(dropout)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.gru:
            self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
            self.rnn = nn.GRU(self.herbert_output_size,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True,
                            dropout = 0 if n_layers < 2 else dropout)
        else:
            self.fc = nn.Linear(self.herbert_output_size, output_dim)
        
        self.lr = lr
        self.training_step_size = lr_decay_step_size
        self.gamma = lr_decay
        self.bert_training = train_herbert
        #set_bert_training(self.bert_training, self.herbert)
        
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.dataset = dataset

        self.model_path = model_path
        self.predicting = predicting
        self.hardware = device
        self.train_adapter = train_adapter
        
        self.metrics = nn.ModuleDict({
            'accuracy': tm.Accuracy()
            # 'recall': tm.Recall(num_classes=output_dim, average='macro'),
            # 'f1': tm.F1Score(num_classes=output_dim, average='macro'),
            # 'precision': tm.Precision(num_classes=output_dim, average='macro')
        })
        self.previous_best_val_accuracy = None

    def forward(self, words):
        if self.predicting:
            tokens = self.tokenizer(list(words), padding = True, truncation = True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.hardware)
            attention_mask = tokens['attention_mask'].to(self.hardware)
            if self.gru:
                embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]
            else:
                embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
        else:
            input_ids = words[1].to(self.hardware)
            attention_mask = words[0].to(self.hardware)
            if self.gru:
                embedded = self.herbert(input_ids, attention_mask = attention_mask)[0]
            else:
                embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
                

        if self.gru:
            _, hidden = self.rnn(embedded)

            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            else:
                hidden = self.dropout(hidden[-1,:,:])

        if self.gru:
            output = self.fc(hidden)
        else:
            output = self.fc(embedded)

        soft = self.softmax(output)

        return soft

    def embed(self, words):
        tokens = self.tokenizer(list(words), padding = True, truncation = True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.hardware)
        attention_mask = tokens['attention_mask'].to(self.hardware)
        embedded = self.herbert(input_ids, attention_mask = attention_mask).pooler_output
        return embedded

    def compute_metrics(self, preds, labels):
        new_metrics = {}
        for name, metric in self.metrics.items():
            new_metrics[name] = metric(
                torch.from_numpy(preds).to(self.hardware), 
                torch.from_numpy(labels).to(self.hardware)
            )
        return new_metrics

    def training_step(self, training_batch, batch_idx):
        x, y = training_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        return dict(
            loss=loss,
            log=dict(
                loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            predictions=logits.detach().cpu().numpy()
        )

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return dict(
            valid_loss=loss,
            log=dict(
                valid_loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            predictions=logits.detach().cpu().numpy()
        )

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return dict(
            test_loss=loss,
            log=dict(
                test_loss=loss.detach()
            ),
            labels=y.detach().cpu().numpy(),
            inputs=x[2],
            predictions=logits.detach().cpu().numpy(),
        )

    def training_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        #metrics = self.compute_metrics(predictions, labels)
        self.optimizer.step()
         
    def validation_epoch_end(self, outputs):
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        print("Validation report")
        print(classification_report(labels,np.argmax(predictions,axis=1)))
        metrics = self.compute_metrics(predictions, labels)
        acc = metrics['accuracy']
        self.log('val_accuracy', acc)
        
        # caching model if some improvement
        if not self.previous_best_val_accuracy or acc > self.previous_best_val_accuracy:
            # save model
            filename = get_pkl_filename(self.model_path, self.dataset)
            if not self.train_adapter:
                torch.save(self.state_dict(), filename)
                print(f"Accuracy improved from {self.previous_best_val_accuracy} to {acc}. Caching model to {filename}\n")
            # save adapters
            if self.train_adapter:
                adapters_dir = filename.split(".")[0] + "_adapters"
                if not os.path.exists(adapters_dir):
                    os.makedirs(adapters_dir)
                self.herbert.save_all_adapters(adapters_dir)
                print(f"Accuracy improved from {self.previous_best_val_accuracy} to {acc}. Caching adapters to {adapters_dir}\n")
            #
            self.previous_best_val_accuracy = acc

        # stopping bert training at some point to only finetune other layers TODO refactor hardcode
        if self.current_epoch >= 7:
            print('Stopped bert training')
            self.bert_training = False
            set_bert_training(False, self)

    def test_epoch_end(self, outputs):
        #avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        labels = np.concatenate([x['labels'] for x in outputs])
        predictions = np.concatenate([x['predictions'] for x in outputs])
        #inputs = np.concatenate([x['inputs'] for x in outputs])
        #self.test_metrics = self.compute_metrics(predictions, labels)

        print("Test report")
        print(classification_report(labels, np.argmax(predictions,axis=1), labels=[0,1,2]))#, target_names=self.target_names))
        print("Confusion matrix")
        print(confusion_matrix(labels, np.argmax(predictions,axis=1), normalize='true'))
        print(np.unique(labels, return_counts=True))
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(labels, np.argmax(predictions,axis=1), normalize='true', ax=ax)
        fig, ax = plt.subplots(figsize=(16, 12))
        plot_confusion_matrix(labels, np.argmax(predictions,axis=1), ax=ax)
        #metrics = self.compute_metrics(predictions, labels)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)#imitation of BertAdam
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.training_step_size, gamma=self.gamma)
        return [self.optimizer], [self.scheduler]

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader
    
    # TODO check what is it used for
    def predictor(self, tokens):
        print(tokens)
        results = []
        logits=self.forward(tokens)
        results.append(logits.cpu().detach().numpy()[0])
        results_array = np.array(results)
        return results_array
