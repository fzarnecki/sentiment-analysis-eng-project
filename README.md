## About the project
The code presented is a refactored repository used for the purpose of an Engineering Project carried out during Data Engineering studies over at Gdansk University of Technology. It features a custom architecture built on top of a BERT model, designed to handle various situations and multiple languages (polish, english, russian).

The solution has been used for:
- Academic Research for the Engineering Project
- Analysis of the Covid-19 pandemic situation on Twitter
- Commercial use in production environment for Voicelab.AI

The code allows to train and use Bert-based models, designed to process textual data and return desired labeling. Here, it is customised to handle Sentiment Analysis with 3 classes (negative, neutral, positive). It is possible to:
- Perform EDA on chosen data ```eda/exploratory_data_analysis.ipynb```
- Clean a dataset (designed for Twitter) ```utils/data_preprocessing.py```
- Train the model on new datasets ```train.py```
- Evaluate performance on test sets ```test.py```
- Label new data, e.g. for the purposes of future analysis ```predict.py```
- Generate intermediate embeddings to obtain a vectorised representation of samples ```generate_embeddings.py```
- Serialize the model for future use ```cache_model.py```

The project has been used to train models used in production environment at Voicelab.AI. The solution shows great capabilities with real-world banking data (evaluation made possible thanks to our cooperation with the company).

## Evaluation results
3 main datasets have been used during the process
- Twitter publications, ~70k
- hand-collected and hand-labeled group of tweets on the topic of Covid-19, ~1k
- private dataset of banking conversations owned by Voicelab.AI, ~700

Each sample has been assigned with a sentiment label (negative, neutral, positive) and each file has been further pre-processed and cleaned.

The model has shown a strong performance on all of the mentioned datasets, especially real-world banking data, proving to be a useful tool not only in academic research, but also in proper data analysis that could be a basis for commercial uses.

| Datasets        | Accuracy           | F1-Score  |Precision  |Recall  |
| :-------------: |:-------------:| :-----:|:-----:|:-----:|
| Twitter      | 0.73 | 0.71 | 0.71 | 0.71 |
| Twitter_covid      | 0.73 | 0.68 | 0.68 | 0.69 |
| Bank      | 0.87      |   0.86 |    0.86 |    0.86 |

## Dataset for training
Data should be saved in .csv format where each utterance is placed in a new line with the following structure:
```
text, sentiment_label 
```
where sentiment_label can take the following values:
- 0 for negative sentiment
- 1 for neutral sentiment
- 2 for positive sentiment

## Training the model
In order to setup model training process 'train.py' file needs to be run, choosing appropriate arguments.

It is necessary to specify 'dataset' one wishes to use and specify neptune project to store training process data. Additionally, one can choose desired gpu to carry training on using the 'gpu' keyword.

Example training command:
```
python3 train.py --dataset /data/training_data.csv --gpu 3 --neptune your-neptune-project
```

It is important to specify a neptune project that one has access to, in order to properly cache training process data.

## Classifying new data
To classify new data using previously trained model it is possible to call labeling.py file with --dataset argument with a path to data one wishes to classify. The data file should contain utterances to classify in csv format, each separated with a newline. Additionally, one needs to specify --model_path argument that contains path to previously cached model weights during training process.

Example labeling command:
```
python3 labeling.py --dataset /data/data_to_label.csv --gpu 3 --model_path models/best_model.ckpt
```
