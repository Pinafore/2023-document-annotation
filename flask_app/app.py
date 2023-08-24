from flask import Flask, request, render_template, url_for, redirect, flash, session, jsonify
from .backend_server import User
# import random
import sqlite3
from .tools import *
import json
from flask_session import Session
# import requests
from datetime import datetime
import os
import threading
import pickle
import time
import csv

Enable_security = True
user_names = ['user1-active', 'user1-LDA', 'user1-SLDA', 'user1-CTM', 'u5', 'u2', 'u3', 'u4']
blocked_users = set()


app = Flask(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
topic_models_dir = os.path.dirname(current_dir)
# all_texts = json.load(open(os.path.join(topic_models_dir,'./Topic_Models/Data/congressional_bill_train.json')))
all_texts = json.load(open(os.path.join(topic_models_dir,'./Topic_Models/Data/congressional_bill_train.json')))
true_labels = list(all_texts['label'].values())
session_stop_events = {}


DATABASE = './database/08_24_pilot/08_24.db'
user_instances = {}


def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        # Create the 'users' table if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         mode INTEGER NOT NULL)''')
        conn.commit()
        # Check if the 'user_id' and 'mode' columns exist in the 'users' table
        cursor.execute("PRAGMA table_info(users)")
        table_info = cursor.fetchall()
        columns = [info[1] for info in table_info]
        
        if 'user_id' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN user_id INTEGER')
            conn.commit()
        
        if 'mode' not in columns:
            cursor.execute('ALTER TABLE users ADD COLUMN mode INTEGER NOT NULL')
            conn.commit()

        cursor.execute('''CREATE TABLE IF NOT EXISTS recommendations
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         label TEXT,
                         doc_id INTEGER,
                         response_time REAL,
                         actual_label TEXT,
                         local_training_acc REAL,
                         purity REAL,
                         local_testing_acc REAL,
                         RI REAL,
                         NMI REAL,
                         test_purity REAL,
                         test_RI REAL,
                         test_NMI REAL,
                         topic_information TEXT,
                         FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        conn.commit()

        cursor.execute('''CREATE TABLE IF NOT EXISTS user_label_track
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         time TEXT,
                         user_labels TEXT,
                         purity REAL,
                         RI REAL,
                         NMI REAL,
                         test_purity REAL,
                         test_RI REAL,
                         test_NMI REAL,
                         click_track TEXT,
                         FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        conn.commit()

        



init_db()

@app.route('/')
def hello():
    return 'Hello World!'

def create_file(filename, columns):
    with open(filename, 'a') as f:
        columns += '\n'
        f.write(columns)


@app.route('/create_user', methods=['POST'])
def create_user():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(user_id) FROM users')
        result = cursor.fetchone()

        if result[0] is not None:
            user_id = result[0] + 1
        else:
            user_id = 1

        mode = user_id % 4

        # print('user id is {}, mode is {}'.format(user_id, mode))
        cursor.execute('INSERT INTO users (user_id, mode) VALUES (?, ?)', (user_id, mode))
        conn.commit()
        user_instances[user_id] = User(mode, user_id)  # Create the User() object and store it in the user_instances dictionary


    filename = './user_data_per_doc/{}.csv'.format(user_id)
    columns = ['user_id', 'doc_id', 'label', 'purity', 'randIndex', 'NMI', 'responseTime', 'sLDAUpFreq' ,'topKeywords', 'recommendID', 'actualLabel']
    create_file(filename, ','.join(columns))

    time_filename = './user_time_data/{}.csv'.format(user_id)
    time_columns = ['user_id', 'minutesPassed', 'Purity', 'randIndex', 'NMI', 'numDocsLabeled', 'sLDAUpFreq']
    create_file(time_filename, ','.join(time_columns))
    
    click_filename = './user_clicks/{}.csv'.format(user_id)
    click_columns = ['user_id', 'recommendID', 'viewID', 'action']
    create_file(click_filename, ','.join(click_columns))

    print('Creating user', user_id)

    
    return jsonify({'user_id': user_id, 'code': 200, 'msg': 'User created'})


def backend_save_user_labels(user_id, time_interval):
    user =  user_instances[user_id]

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            user_labels = user.alto.user_labels
            dict_as_string = json.dumps(user_labels)
            clicks = json.dumps(user.click_tracks)
            purity, RI, NMI, user_purity, user_RI, user_NMI = user.get_metrics_to_save()
            cursor.execute('''INSERT INTO user_label_track
                            (user_id, time, user_labels, purity, RI, NMI, test_purity, test_RI, test_NMI, click_track)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (user_id, time_interval, dict_as_string, purity, RI, NMI, user_purity, user_RI, user_NMI, clicks))
            conn.commit()

            result = {}
            result['code'] = 200
            result['msg'] = 'SUCCESS'
            # conn.close()
            print('inserted')
            return result
        else:
            return {"code": 404, "msg": "User not found"}

@app.route('/recommend_document', methods=['POST'])
def nist_recommend():
    user_id = request.json.get('user_id')
    label = request.json.get('label')
    doc_id = request.json.get('document_id')
    response_time = request.json.get('response_time')

    user =  user_instances[user_id]
    ltr, lte, purity, RI, NMI, user_purity, user_RI, user_NMI, result = user.round_trip1(label, doc_id, response_time)
    document_topics = user.get_doc_information_to_save(doc_id)
    

    actual_label = true_labels[int(doc_id)]

    with open('./user_data_per_doc/{}.csv'.format(user_id), 'a') as f:
        try:
            data_string = ','.join([
                    str(user_id),
                    str(doc_id),
                    str(label),
                    str(purity),
                    str(RI),
                    str(NMI),
                    str(response_time),
                    str(user.slda_update_freq),
                    str(document_topics['topics'][0][0]),
                    str(user.alto.last_recommended_doc_id),
                    actual_label]) + '\n'
            f.write(data_string)
        except:
            data_string = ','.join([
                    str(user_id),
                    str(doc_id),
                    str(label),
                    str(purity),
                    str(RI),
                    str(NMI),
                    str(response_time),
                    str(user.slda_update_freq),
                    'None',
                    str(user.alto.last_recommended_doc_id),
                    actual_label]) + '\n'
            f.write(data_string)
            print('failed to save')
        
        with open('./user_clicks/{}.csv'.format(user_id), 'a') as f:
            data_string = ','.join([
                    str(user_id),
                    str(user.alto.last_recommended_doc_id),
                    str(doc_id),
                    'label']) + '\n'
                
            # print('data string is ', data_string)
            f.write(data_string)

    document_topics = json.dumps(document_topics)

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            cursor.execute('''INSERT INTO recommendations
                            (user_id, label, doc_id, response_time, actual_label, local_training_acc, local_testing_acc, purity, RI, NMI, test_purity, test_RI, test_NMI,topic_information)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (user_id, label, doc_id, response_time, actual_label, ltr, lte, purity, RI, NMI, user_purity, user_RI, user_NMI, document_topics))
            conn.commit()

            result['code'] = 200
            result['msg'] = 'SUCCESS'
            # conn.close()
            return jsonify(result)
        else:
            return jsonify({"code": 404, "msg": "User not found"})


@app.route('/skip_document', methods=['POST'])
def skip_document():
    user_id = request.json.get('user_id')
    user =  user_instances[user_id]
    next_doc_id = user.skip_doc()
            # conn.close()
    return jsonify({'document_id': str(next_doc_id)})
        
 
@app.route('/get_topic_list', methods=['POST'])
def get_list():
    user_id = request.json.get('user_id')
    recommend_action = True
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            user =  user_instances[user_id]
            result = user.get_document_topic_list(recommend_action)
            result['code'] = 200
            result['msg'] = 'SUCCESS'
            return jsonify(result)
        else:
            return jsonify({"code": 404, "msg": "User not found"})


def check_active_list(user_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            user =  user_instances[user_id]
            result = user.check_active_list()
            result['code'] = 200
            result['msg'] = 'SUCCESS'
            return result
        else:
            return {"code": 404, "msg": "User not found"}

@app.route('/get_document_information', methods=['POST'])
def get_doc_info():
    user_id = request.json.get('user_id')
    doc_id = request.json.get('document_id')
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            user =  user_instances[user_id]
            result = user.get_doc_information(doc_id)
            result['code'] = 200
            result['msg'] = 'SUCCESS'
            # conn.close()
            with open('./user_clicks/{}.csv'.format(user_id), 'a') as f:
                data_string = ','.join([
                        str(user_id),
                        str(user.alto.last_recommended_doc_id),
                        str(doc_id),
                        'click']) + '\n'
                
                # print('data string is ', data_string)
                f.write(data_string)
            return jsonify(result)
        else:
            return jsonify({"code": 404, "msg": "User not found"})


def record_data_for_session(session_id, stop_event):
    counter = 1
    while not stop_event.is_set():
        time.sleep(60)  # Wait for 60 seconds
        # Your logic to record data for the session
        user =  user_instances[session_id]
        with open('./user_time_data/{}.csv'.format(session_id), 'a') as f:
            data_string = ','.join([
                    str(session_id),
                    str(counter),
                    str(user.purity), 
                    str(user.RI), 
                    str(user.NMI), 
                    str(user.alto.num_docs_labeled),
                    str(user.slda_update_freq)]) + '\n'
            
            # print('data string is ', data_string)
            f.write(data_string)
        
        with open('./user_clicks/{}.csv'.format(session_id), 'a') as f:
            data_string = ','.join([
                str(session_id),
                str(user.alto.last_recommended_doc_id),
                'None',
                'None']) + '\n'
                
            # print('data string is ', data_string)
            f.write(data_string)
        print(f"Recording data for session: {session_id}")
        counter += 1

'''
This is frontend session
'''
    

@app.route("/save_user_label", methods=["POST"])
def save_user_labels():
    print('saving labels...')
    user_id = session.get("user_id")
    label_seconds = request.get_data(as_text=True)
    result = backend_save_user_labels(user_id, label_seconds)

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True)