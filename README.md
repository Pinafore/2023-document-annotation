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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pinafore/2023-document-annotation/blob/working-app/synthetic_experiment.ipynb)


## Pipeline

1. Data Preprocessing (Tokenizing words, and save tokenized passages as a pickle file. A new copy of the json file with any passsages that have length 0 after tokenization will be removed and saved)

      To process a dataset, first please provide a dataset in json format that could be read by [pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)

      The table must be contain the following columns

      | text | label |  sub_labels |
    | --------------- | --------------- | --------------- |
    | Text Passages    |  Major Topic | Minor Topic |
   
    
    To process the data, run the following. To process your own dataset, run `python data_process.py --help` to see the arguments
  ```
  cd Topic_Models
  python data_process.py 
  ```


2. Topic Model Training

   The following are the list of topic models you can train
   LDA, supervised LDA, embedded topic model, contextualized topic model, bertopic, partially labaled LDA, labeled LDA

   To train a topic model, run
```
cd Topic_Models

python train_save_topic_model.py --num_topics <number_of_topics> \ 
--num_iters <number_of_training_iterations> \
--model_type <LDA_or_SLDA_or_ETM_or_CTM_or_Bertopic_or_PLDA_or_LLDA> \
--load_data_path <Processed_pickle_data_path>
--num_topics <number_of_topics>
--raw_text_path <json_file_path_filtered_by_data_process.py>
```

   The trained topic models will be saved to './Topic_Models/Model/' directory. 
   The saved models will be in a pickle format '{model_type}_{number of topics}.pkl'

    

3. To reproduce the synthetic experiment, first train your model with step 1 or 2.
   Then open synthetic_experiment.ipynb to run the models

4. To plot the results for the synthetic experiment, go to new_model_plot.ipynb'. Then
   read the saved results and plot
