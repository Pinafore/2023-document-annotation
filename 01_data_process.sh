#!/bin/bash

# First process the data
python ./flask_app/data_process.py \
       --doc_dir /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/gtd/GTD.json \
       --save_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/gtd/GTD_processed.pkl \
       --new_json_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/gtd/new_GTD.json



python ./flask_app/data_process.py \
       --doc_dir /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/community_resilience/nist_disaster_tweets.json \
       --save_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/community_resilience/nist_disaster_tweets_processed.pkl \
       --new_json_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/community_resilience/processed_nist_disaster_tweets.json