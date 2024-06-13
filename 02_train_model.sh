#!/bin/bash

python ./flask_app/Topic_Models/train_models.py \
        --num_topics 35 \
        --num_iters 3000 \
        --model_type LDA \
        --load_data_path ./flask_app/Data/community_resilience/community_resilience_processed.pkl\
        --raw_text_path ./flask_app/Data/community_resilience/community_resilience_clean.json\
        --save_trained_model_path ./flask_app/Topic_Models/trained_models/LDA_35.pkl

python ./flask_app/Topic_Models/train_models.py \
        --num_topics 35 \
        --num_iters 3000 \
        --model_type SLDA \
        --load_data_path ./flask_app/Data/community_resilience/community_resilience_processed.pkl\
        --raw_text_path ./flask_app/Data/community_resilience/community_resilience_clean.json\
        --save_trained_model_path ./flask_app/Topic_Models/trained_models/SLDA_35.pkl

python ./flask_app/Topic_Models/train_models.py \
        --num_topics 35 \
        --num_iters 250 \
        --model_type CTM \
        --load_data_path ./flask_app/Data/community_resilience/community_resilience_processed.pkl\
        --raw_text_path ./flask_app/Data/community_resilience/community_resilience_clean.json\
        --save_trained_model_path ./flask_app/Topic_Models/trained_models/CTM_35.pkl

python ./flask_app/Topic_Models/train_models.py \
        --num_topics 200 \
        --num_iters 5000 \
        --model_type LDA \
        --load_data_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/gtd/GTD_processed.pkl\
        --raw_text_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/gtd/new_GTD.json\
        --save_trained_model_path ./flask_app/Topic_Models/trained_models/LDA_200.
        
python ./flask_app/Topic_Models/train_models.py \
        --num_topics 35 \
        --num_iters 3000 \
        --model_type LDA \
        --load_data_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/community_resilience/nist_disaster_tweets_processed.pkl\
        --raw_text_path /srv/www/active-topic-modeling/cleaned_up_code/camera_ready_copy/flask_app/Data/community_resilience/processed_nist_disaster_tweets.json\
        --save_trained_model_path ./flask_app/Topic_Models/trained_models/nist_disaster/LDA_35.pkl