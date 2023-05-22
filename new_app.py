from flask import Flask, request, render_template, url_for, redirect, flash, session, jsonify
import requests
from backend_server import User
import random
import sqlite3
import os
from community_resilience.tools import *
import time
import json

app = Flask(__name__)

DATABASE = 'local_users.db'
DATABASE = 'beta_users.db'

user_instances = {}
MODES = [0, 1, 2, 3]
# MODES = [1, 1, 1, 1]
# MODES = [2, 2, 2, 2]
# MODES = [3, 3, 3, 3]
# MODES = [0, 3, 3, 3]
# MODES = [0, 0, 0, 0]
GLOBAL_COUNTER = 0

def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    conn = create_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     mode INTEGER NOT NULL);''')
    conn.execute('''CREATE TABLE IF NOT EXISTS recommendations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id INTEGER,
                     user_name TEXT,
                     label TEXT,
                     doc_id INTEGER,
                     response_time REAL,
                     actual_label TEXT,
                     local_training_acc REAL,
                     global_training_acc REAL,
                     local_testing_acc REAL,
                     global_testing_acc REAL,
                     FOREIGN KEY (user_id) REFERENCES users (id));''')
    conn.close()


    return 'Hello World!'

# @app.route('/create_user', methods=['POST'])
def create_user(user_name):
    global GLOBAL_COUNTER
    mode_idx = GLOBAL_COUNTER%4
    mode = MODES[mode_idx]
    GLOBAL_COUNTER += 1
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (mode, user_name) VALUES (?, ?)', (mode, user_name))
        conn.commit()
        cursor.execute('SELECT last_insert_rowid()')
        result = cursor.fetchone()
        user_id = result[0]
        user_instances[user_id] = User(mode)  # Create the User() object and store it in the user_instances dictionary
    
    print('creating user ', user_id)
    return {'user_id': user_id, 'code': 200, 'msg': 'User created'}

# @app.route('/recommend_document', methods=['POST'])
def nist_recommend(user_id, label, document_id, response_time, actual_label):
    conn = create_connection()
    cursor = conn.execute('SELECT mode FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()

    if row:
        mode = row[0]
        user =  user_instances[user_id]
        ltr, lte, gtr, gte, result = user.round_trip1(label, document_id, response_time)

        # Save the label, doc_id, and response_time for the current user_id

        conn.execute('INSERT INTO recommendations (user_id, label, doc_id, response_time, actual_label, local_training_acc, local_testing_acc, global_training_acc, global_testing_acc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (user_id, label, document_id, response_time, actual_label, ltr, lte, gtr, gte))
        conn.commit()

        result['code'] = 200
        result['msg'] = 'SUCCESS'
        conn.close()
        return jsonify(result)
    else:
        conn.close()
        return jsonify({"code": 404, "msg": "User not found"})

# @app.route('/get_topic_list', methods=['POST'])
def get_list(user_id):
    # user_id = request.json.get('user_id')

    conn = create_connection()
    cursor = conn.execute('SELECT mode FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    if row:
        # mode = row[0]
        user =  user_instances[user_id]
        result = user.get_document_topic_list()
        result['code'] = 200
        result['msg'] = 'SUCCESS'
        conn.close()
        return jsonify(result)
    else:
        conn.close()
        return jsonify({"code": 404, "msg": "User not found"})

# @app.route('/get_document_information', methods=['POST'])
def get_doc_info(user_id, doc_id):
    print('fetching document information')
    # user_id = request.json.get('user_id')
    # doc_id = request.json.get('document_id')

    conn = create_connection()
    cursor = conn.execute('SELECT mode FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()
    if row:
        # mode = row[0]
        user =  user_instances[user_id]
        result = user.get_doc_information(doc_id)
        # print(result)
        result['code'] = 200
        result['msg'] = 'SUCCESS'
        conn.close()
        # print('SUCCESS')
        return jsonify(result)
    else:
        conn.close()
        return jsonify({"code": 404, "msg": "User not found"})



'''
This is frontend session
'''

init_db()
all_texts = json.load.open('./Data/Nist_all_labeled.json')
true_labels = all_texts['label']

app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"


@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method =="POST":
        name = request.form["name"]
        global GLOBAL_COUNTER
        mode_idx = GLOBAL_COUNTER%4
        mode = MODES[mode_idx]
        GLOBAL_COUNTER += 1

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (mode, user_name) VALUES (?, ?)', (mode, name))
            conn.commit()
            cursor.execute('SELECT last_insert_rowid()')
            result = cursor.fetchone()
            user_id = result[0]
            security_code = os.urandom(24).hex()
            # user_instances[user_id] = {}  # Create the User() object and store it in the user_instances dictionary
            user_instances[user_id] = User(mode)
            session["user_id"] = user_id


        print('creating user ', user_id)


        return redirect(url_for("home_page", name=name, user_id=user_id))
    
    return render_template("login.html")



# READING WHAT THE STUDY IS ABOUT
@app.route("//firstpage//<name>/", methods = ["POST", "GET"])
def home_page(name):
    # print(session)
    # if session.get("name") != name:
    #     # if not there in the session then redirect to the login page
    #     return redirect("/login")
    if request.method =="POST":
        return redirect(url_for("active_check", name=name))
        
    return render_template("first.html", name=name)

@app.route("//checkactive//<name>//", methods=["POST", 'GET'])
def active_check(name):
    # if session.get("name") != name:
    #     # if not there in the session then redirect to the login page
    #     return redirect("/login")

    # get_topic_list = get_list(session['user_id'])
        # print(session)
    user_id = session['user_id']
    
    if user_id % 4 == 0:
        return redirect(url_for("active_list", name=name))

    else:
        return redirect(url_for("non_active_list", name=name))
    

@app.route("//active_list//<name>", methods=["POST", "GET"])
def active_list(name):
    # if session.get("name") != name:
    # # if not there in the session then redirect to the login page
    #     return redirect("/login")

    # get_topic_list = url + "//get_topic_list"
        # print(session)
    # topics = requests.post(get_topic_list, json={
    #                         "user_id": session['user_id']
    #                         }).json()
    
    topics = get_list(session['user_id'])

    rec = str(topics["document_id"])
    print(topics["keywords"])

    results = get_single_document(topics["cluster"]["1"], all_texts)
    # print(results)

    # results = get_texts(topic_list=topics, all_texts=all_texts)
    # results = results["1"]

    if request.method =="POST":
        return redirect(url_for("finish"))
 
    return render_template("active_list.html", results=results, name=name, rec = rec)

@app.route("//non_active_list//<name>", methods=["POST", "GET"])
def non_active_list(name):
    # if session.get("name") != name:
    #     # if not there in the session then redirect to the login page
    #     return redirect("/login")

    topics = get_list(session['user_id'])

    recommended = int(topics["document_id"])

    results = get_texts(topic_list=topics, all_texts=all_texts)


    # This part can be directly used from my functions
    sliced_results = get_sliced_texts(topic_list=topics, all_texts=all_texts)
    # print(sliced_results)

    keywords = topics["keywords"] 
    # print(keywords)

    if request.method =="POST":
        return redirect(url_for("finish"))

    return render_template("nonactive.html",sliced_results=sliced_results, results=results, name=name, keywords=keywords, recommended=recommended, document_list = topics["cluster"])

@app.route("//acitve//<name>//<document_id>/", methods=["GET", "POST"])
def active(name, document_id):
    topics = get_list(session['user_id'])
    print(topics.keys())

    # results = get_texts(topic_list=topics, all_texts=all_texts)
    text = all_texts["text"][str(document_id)]
    st =time.time()

    user_labels = user_instances[session['user_id']].user_labels
    # old_labels = list(set(predictions))
    old_labels = list(user_labels)

    print(old_labels)
    if request.method =="POST":
        name=name
        document_id=document_id
        user_id = session["user_id"]
        et = time.time()
        response_time = st- et
        label = request.form.get("label").replace(' ', '').lower()

        save_response(name, label, response_time, document_id, user_id)
        # recommend_document = url + "//recommend_document"
        

        result = nist_recommend(user_id, label, document_id, response_time, true_labels[str(document_id)])

        
        if result['code'] == 200:
            next = result['document_id']

            return redirect(url_for("active", name=name, document_id=next, predictions=old_labels))
        else:
            return render_template("activelearning.html", text =text, predictions=old_labels ) 
            # return jsonify({"code": 404, "msg": "User not found"})

  
 
    return render_template("activelearning.html", text =text, predictions=old_labels ) 


@app.route("//get_label//<document_id>//", methods=["POST", 'GET'])
def get_label(document_id):
    document_id = document_id 
    user_id=session["user_id"] 
    # get_document_information = url + "//get_document_information"
    # data = requests.post(get_document_information, json={ "document_id": document_id,
    #                                                     "user_id":user_id
    #                                                      }).json()
    data = get_doc_info(user_id, document_id)                                               


    return redirect( url_for("label", response=data, name=session["name"], document_id=document_id))


@app.route("/logout", methods=["POST", "GET"])
def finish():
    name = session['name']
    session.pop(name, None)
    return redirect(url_for("login"))

@app.before_request
def require_login():
    allowed_route = ['login']
    if request.endpoint not in allowed_route and "name" not in session:
        return redirect(url_for("login"))
    
@app.route('//non_active_label//<name>//<document_id>/', methods=["POST", "GET"])
def non_active_label(name, document_id):
    st = time.time()
    # get_document_information = url + "//get_document_information"
    # response = requests.post(get_document_information, json={ "document_id": document_id,
    #                                                     "user_id":session["user_id"]
    #                                                      }).json()

    user_id = session[user_id]
    response = get_doc_info(user_id, document_id) 

    text = all_texts["text"][str(document_id)]
    words = get_words(response["topic"],  text)
    user_labels = user_instances[user_id].user_labels
    
    old_labels = list(user_labels)

    if request.method =="POST":
        name=name 
        document_id=str(document_id)
        user_id = session["user_id"]
        et = time.time()
        response_time = et - st
        label = request.form.get("label")
        # recommend_document = "https://nist-topic-model.umiacs.umd.edu/recommend_document"
        # recommend_document = url + "//recommend_document"
        # posts = requests.post(recommend_document, json={
        # "user_id" : int(user_id),
        # "label": label,
        # "response_time": response_time,
        # "document_id" : document_id
        # }).json()

        posts = nist_recommend(user_id, label, document_id, response_time, true_labels[str(document_id)])


        # print(posts.keys())
        next = posts["document_id"]
        # predictions.append(label.lower())
        # old_labels = list(set(predictions))
        # print(old_labels)

        save_response(name, label, response_time, document_id, user_id)
        # get_document_information = url + "//get_document_information"
        # response = requests.post(get_document_information, json={ "document_id": posts["document_id"],
        #                                                 "user_id":session["user_id"]
        #                                                  }).json()
        
        response = get_doc_info(user_id, next)
        # print(response["prediction"]) 
        return redirect(url_for("non_active_label", response=response, words=words, document_id=posts["document_id"], name=name, predictions=old_labels, pred=response["prediction"]))

    return render_template("nonactivelabel.html", response=response, words=words, document_id=document_id, text=text, name=name, predictions=old_labels, pred=response["prediction"])

@app.route("/non_active/<name>/<topic_id>//<documents>")
def topic(name, topic_id, documents):
    print(topic_id)
    # res = get_single_document(documents, all_texts)
    # print(res)
    res = get_single_document(documents.strip("'[]'").split(", "), all_texts)

    return  render_template("topic.html", res = res, topic_id=topic_id)  