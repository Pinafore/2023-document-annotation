# Document Annotation App
This is a single flask app interface for speeding up document annotation and testing the effectiveness of Active Learning with Topic Overview on labeled datasets


## Run Locally

```
flask run -p <your port number>
```



Datase Information

```
20newsgroup

congressional_bill_project_dataset

Dataset: Congressional Bill dataset from the following websites

https://www.comparativeagendas.net/us

http://www.congressionalbills.org

Code book to match topics: https://comparativeagendas.s3.amazonaws.com/codebookfiles/Codebook_PAP_2019.pdf
```

## Pipeline

1. Data Preprocessing (Tokenizing words, and save tokenized passages as a pickle file. A new copy of the json file with any passsages that have length 0 after tokenization will be removed and saved)

  To process a dataset, please provide a dataset 


2. Topic Model training


   To train a classical topic model, run
```
python create_classical_model.py --num_topics <number_of_topics> \ 

--num_iters <number_of_training_iterations> \

--model_type <LDA_or_SLDA> \

--load_data_path <your_processed_data_path>

--train_len <number_of_texts_to_train>
```

Model not updated yet. 
More Details will be added later

