# TENOR: Efficient Document Labeling Tool to Exploratory Data Analysis

:fire:
TENOR is a user interface for speeding up document labeling process and reducing the number of documents needed to be labeled. See the paper for details:
- Poursabzi-Sangdeh, Forough, et al. (2016). Alto: Active learning with topic overviews for speeding label induction and document labeling. ACL. https://aclanthology.org/P16-1110.pdf


## References

If you find this tool helpful, you can cite the following paper:

```bibtex
@inproceedings{poursabzi2016alto,
  title={Alto: Active learning with topic overviews for speeding label induction and document labeling},
  author={Poursabzi-Sangdeh, Forough and Boyd-Graber, Jordan and Findlater, Leah and Seppi, Kevin},
  booktitle={Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1158--1169},
  year={2016}
}
```


## Getting Started

This tool's frontend code interface is adopted from [this repository](https://github.com/daniel-stephens/community_resilience). To run the app interface locally with the default Bills dataset, follow these steps:


```bash
git clone https://github.com/daniel-stephens/community_resilience.git
cd 2023-document-annotation
pip install -r requirements.txt
```

## Setup
Preprocess the data for topic model training. The processed data will be saved to the specified --new_json_path directory
```
./01_data_process.sh
```

Train topic models Download trained topic models from mywebs.com Or train your own models locally with the following script
```
./02_train_model.sh
```

Note: Models will be saved to the save_trained_model_path. Three trained models on the bills dataset are in [here](https://drive.google.com/drive/folders/1-k6YcC2KLp8iULGF5zmpAYlpk49dbX4W?usp=sharing). If downloading a trained model, place it in the ./flask_app/Topic_Models/trained_Models directory. The default number of topics loaded for this app is 35. If you wish to use a different number of topics, train the topic models accordingly, and update line 128 in app.py to reflect the desired number of topics.

Run the web application
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

## Contributing

We welcome contributions to this project. If you have suggestions for improving the tool or would like to contribute code, documentation, or bug reports, please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a new Pull Request

## Acknowledgements

This project was built upon the work of the following repositories:
- [Contextualized Topic Models](https://github.com/MilaNLProc/contextualized-topic-models)
- [Tomotopy](https://github.com/bab2min/tomotopy)

We extend our gratitude to the authors of these original repositories for their valuable contributions and inspiration.

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE file for details.

## Contact

For any additional questions or comments, please contact [zli12321@umd.edu].

Thank you for using or contributing to the Document Annotation NLP Tool!
