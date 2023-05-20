from flask import Flask, request, render_template, url_for, redirect, flash, session, jsonify
import requests
from backend_server import User
import random
import sqlite3
import os
from community_resilience.tools import *


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
def nist_recommend():
    user_id = request.json.get('user_id')
    label = request.json.get('label')
    doc_id = request.json.get('document_id')
    response_time = request.json.get('response_time')
    actual_label = request.json.get('actual_label')

    conn = create_connection()
    cursor = conn.execute('SELECT mode FROM users WHERE id = ?', (user_id,))
    row = cursor.fetchone()

    if row:
        mode = row[0]
        user =  user_instances[user_id]
        ltr, lte, gtr, gte, result = user.round_trip1(label, doc_id, response_time)

        # Save the label, doc_id, and response_time for the current user_id

        conn.execute('INSERT INTO recommendations (user_id, label, doc_id, response_time, actual_label, local_training_acc, local_testing_acc, global_training_acc, global_testing_acc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                     (user_id, label, doc_id, response_time, actual_label, ltr, lte, gtr, gte))
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
def get_doc_info():
    print('fetching document information')
    user_id = request.json.get('user_id')
    doc_id = request.json.get('document_id')

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
all_texts = './Data/Nist_all_labeled.json'

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