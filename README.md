# TENOR: Efficient Document Labeling Tool to Exploratory Data Analysis

:fire:
TENOR is a user interface for speeding up document labeling process and reducing the number of documents needed to be labeled. See the paper for details:
- Poursabzi-Sangdeh, Forough, et al. (2016). Alto: Active learning with topic overviews for speeding label induction and document labeling. ACL. https://aclanthology.org/P16-1110.pdf


## Getting Started

This tool's frontend code interface is adopted from [this repository](https://github.com/daniel-stephens/community_resilience). To run the app interface locally with the default Bills dataset, follow these steps:


```bash
git clone https://github.com/daniel-stephens/community_resilience.git
cd 2023-document-annotation
pip install -r requirements.txt
```

# Setup
Preprocess the data for topic model training. The processed data will be saved to the specified --new_json_path directory
```
./01_data_process.sh
```

Train topic models Download trained topic models from mywebs.com Or train your own models locally with the following script
```
./02_train_model.sh
```

Note: Models will be saved to the save_trained_model_path. If downloading a trained model, place it in the ./flask_app/Topic_Models/trained_Models directory. The default number of topics loaded for this app is 35. If you wish to use a different number of topics, train the topic models accordingly, and update line 128 in app.py to reflect the desired number of topics.

# Run the web application
```
./03_run_app.sh
```

Then, open a browser and go to localhost:5050 or specify the port you want.


## Dataset Information

This app supports two datasets:

1. **20newsgroup**: A collection of newsgroup documents.
2. **congressional_bill_project_dataset**: A compilation of Congressional Bill documents.

For the Congressional Bill dataset, the app utilizes data from these sources:

- [Comparative Agendas Project](https://www.comparativeagendas.net/us)
- [Congressional Bills](http://www.congressionalbills.org)

To correlate topics with labels, consult the [Codebook](https://comparativeagendas.s3.amazonaws.com/codebookfiles/Codebook_PAP_2019.pdf).
