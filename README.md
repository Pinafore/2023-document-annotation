# Document Annotation App
This is a single flask app interface for speeding up document annotation and testing the effectiveness of Active Learning with Topic Overview on labeled datasets


## Run Interface Locally With Default Congressional Bill Dataset

```
cd flask_app
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

      To process a dataset, first please provide a dataset in json format that could be read by [pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)

      The table must be contain the following columns

      | text | label |  sub_labels |
    | --------------- | --------------- | --------------- |
    | Text Passages    |  Major Topic | Minor Topic |
   
    
    To process the data, run the following
  ```
  cd Topic_Models
  python data_process.py --doc_dir <json_file_path> --save_path <pickle_file_name_to_be_saved>
  ```


2. Classical Topic Model Training


   To train a classical topic model, run
```
python create_classical_model.py --num_topics <number_of_topics> \ 

--num_iters <number_of_training_iterations> \

--model_type <LDA_or_SLDA> \

--load_data_path <your_processed_data_path>

--train_len <number_of_texts_to_train>
```

3. Neural Topic Model Training (we use embedded neural topic model here)

    To train a neural topic model, run
    
```
python neural_model.py --num_topics <number_of_topics> \ 

--num_iters <number_of_training_iterations> \

--model_type ETM \

--load_data_path <your_processed_data_path>

--train_len <number_of_texts_to_train>
```
    

Model not updated yet. 
More Details will be added later
