from flask import Flask, request, render_template, url_for, redirect, flash, session, jsonify
from backend_server import User
import random
import sqlite3
from tools import *
import json
from flask_session import Session
import requests
from datetime import datetime
import os

app = Flask(__name__)
all_texts = json.load(open("./Data/CongressionalBill/congressional_bills.json"))
true_labels = list(all_texts['label'].values())

DATABASE = 'local_users.db'
DATABASE = 'server_users.db'
DATABASE = 'beta_testing.db'
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

init_db()

@app.route('/')
def hello():
    return 'Hello World!'

# @app.route('/create_user', methods=['POST'])
def create_user():
    global GLOBAL_COUNTER
    mode_idx = GLOBAL_COUNTER%4
    mode = MODES[mode_idx]
    GLOBAL_COUNTER += 1
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (mode) VALUES (?)', (mode,))
        conn.commit()
        cursor.execute('SELECT last_insert_rowid()')
        result = cursor.fetchone()
        user_id = result[0]
        user_instances[user_id] = User(mode)  # Create the User() object and store it in the user_instances dictionary
    
    print('creating user ', user_id)
    return {'user_id': user_id, 'code': 200, 'msg': 'User created'}

# @app.route('/recommend_document', methods=['POST'])
def nist_recommend(user_id, label, doc_id, response_time):
    # user_id = request.json.get('user_id')
    # label = request.json.get('label')
    # doc_id = request.json.get('document_id')
    # response_time = request.json.get('response_time')
    # actual_label = request.json.get('actual_label')
    # print('doc id is {}'.format(doc_id))
    actual_label = true_labels[int(doc_id)]

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
        # return jsonify(result)
        return result
    else:
        conn.close()
        # return jsonify({"code": 404, "msg": "User not found"})
        return {"code": 404, "msg": "User not found"}

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
        # return jsonify(result)
        return result
    else:
        conn.close()
        # return jsonify({"code": 404, "msg": "User not found"})
        return {"code": 404, "msg": "User not found"}

# @app.route('/get_document_information', methods=['POST'])
def get_doc_info(doc_id, user_id):
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
        # return jsonify(result)
        return result
    else:
        conn.close()
        # return jsonify({"code": 404, "msg": "User not found"})
        return {"code": 404, "msg": "User not found"}



'''
This is frontend session
'''

predictions = []
os.urandom(24).hex()


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/")
def home():
    return redirect(url_for("login"))


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method =="POST":
        with open('./static/users/users.json') as user_file:
            name_string = user_file.read()
            names = json.loads(name_string)
        name = request.form["name"]
        session["name"] = name
        session["start_time"] = ""
        
        

        if name in list(names.keys()):
            session["name"] = names[name]['username']
            session["labels"] = names[name]["labels"]
            session["user_id"] = names[name]["id"]
            session["labelled_document"] = names[name]["labelled_document"]
            
            user_id = session["user_id"]
            print('user id is {}'.format(user_id))
            return redirect(url_for("active_check", name=name))
            
        else:
            # user = requests.post(url + "//create_user", {"user_session": name})
            user = create_user()
            user_id = user["user_id"] 
            session["user_id"] = user_id
            session["labels"] = ""
            session["labelled_document"] = ""
            data = {
                "username": name, 
                "id" : user_id,
                "labels" : session["labels"],
                "labelled_document" : session["labelled_document"]
            }

            session["user_id"] = user_id  
            names[name]=data
            
            with open('./static/users/users.json', mode='w', encoding='utf-8') as name_json:
                # names = json.loads(name_string)
                names[name] = data
                json.dump(names, name_json, indent=4)

            
            
            return redirect(url_for("home_page", name=name, user_id=user_id))
    return render_template("login.html") 


# READING WHAT THE STUDY IS ABOUT
@app.route("//firstpage//<name>/", methods = ["POST", "GET"])
def home_page(name):
    if session.get("name") != name:
        # if not there in the session then redirect to the login page
        return redirect("/login")
    if request.method =="POST":
        return redirect(url_for("active_check", name=name))
    return render_template("first.html", name=name)

@app.route("//checkactive//<name>//", methods=["POST", 'GET'])
def active_check(name):
    if session.get("name") != name:
        # if not there in the session then redirect to the login page
        return redirect("/login")


    topics = get_list(session['user_id'])

    print(topics)
    if len(topics["cluster"].keys()) == 1:
        return redirect(url_for("active_list", name=name))
    else:
        return redirect(url_for("non_active_list", name=name))
    
@app.route("//active_list//<name>", methods=["POST", "GET"])
def active_list(name):
    if session.get("name") != name:
    # if not there in the session then redirect to the login page
        return redirect("/login")

    # get_topic_list = url + "//get_topic_list"
    # topics = requests.post(get_topic_list, json={
    #                         "user_id": session['user_id']
    #                         }).json()

    topics = get_list(session['user_id'])
    rec = str(topics["document_id"])
    docs = list(set(session["labelled_document"].strip(",").split(",")))
    # print(docs)
    docs_len= len(docs)

    results = get_single_document(topics["cluster"]["1"], all_texts, docs)

    if request.method =="POST":
        return redirect(url_for("finish"))
    return render_template("active_list.html", results=results, name=name, rec = rec, docs_len=docs_len)

@app.route("//non_active_list//<name>", methods=["POST", "GET"])
def non_active_list(name):
    if session.get("name") != name:
        # if not there in the session then redirect to the login page
        return redirect("/login")

    # get_topic_list = url + "//get_topic_list" 
    topics = get_list(session['user_id'])
        # print(session)

    recommended = int(topics["document_id"])

    docs = list(set(session["labelled_document"].strip(",").split(",")))
    docs_len = len(docs)
    print(recommended)

    recommended_topic, recommended_block = get_recommended_topic(recommended, topics, all_texts)
    print(recommended_block)

    results = get_texts(topic_list=topics, all_texts=all_texts, docs=docs)

    sliced_results = get_sliced_texts(topic_list=topics, all_texts=all_texts, docs=docs)
    # print(sliced_results)
 
    keywords = topics["keywords"] 
    # print(keywords)

    if request.method =="POST":
        return redirect(url_for("finish"))

    return render_template("nonactive.html", sliced_results=sliced_results, results=results, name=name, keywords=keywords, recommended=str(recommended), document_list = topics["cluster"], docs_len = docs_len, recommended_block=recommended_block, recommended_topic=recommended_topic)


 
@app.route("//active//<name>//<document_id>/", methods=["GET", "POST"])
def active(name, document_id):
    topics = get_list(session['user_id'])

    text = all_texts["text"][str(document_id)]

    session["start_time"] = str(session["start_time"]) + "+" + str(datetime.now().strftime("%H:%M:%S"))

    labels = list(set(session["labels"].strip(",").split(",")))
    docs = list(set(session["labelled_document"].strip(",").split(",")))
    docs_len = len(docs)
    total = len(all_texts["text"])

    if request.method =="POST":
        label = request.form.get("label").lower()
        drop = request.form.get("suggestion").lower()

        label = str(drop) + str(label)
        name=name
        document_id=document_id
        user_id = session["user_id"]

        st = datetime.strptime(session["start_time"].strip("+").split("+")[-2], "%H:%M:%S")
        et = datetime.strptime(session["start_time"].strip("+").split("+")[-1], "%H:%M:%S")

        response_time = str(et-st)
        
        
        save_response(name, label, response_time, document_id, user_id)
        

        posts = nist_recommend(int(user_id), label, document_id, response_time)

        next = posts["document_id"]
        # print(posts.keys())
        predictions.append(label.lower()) 
        
        # session["labelled_document"] = session["labelled_document"]+","+str(document_id)
        session["labels"] = session["labels"] + "," + label
        labels = list(set(session["labels"].strip(",").split(",")))
        print(labels)
        session["labelled_document"] = session["labelled_document"]+","+str(document_id)
        # print(session)
        # print([x.strip("") for x in session["labels"].split(",")])
        # print([x.strip("") for x in session["labelled_document"].split(",")])
        save_labels(session)
        docs = list(set(session["labelled_document"].strip(",").split(",")))
        docs_len = len(docs)
        # print(label) 
        flash("Response has been submitted")
        # print('waiting to get to the next page')
        return redirect(url_for("active", name=name, document_id=next))
        # return redirect(url_for("active", name=name, document_id=next, predictions=labels, docs_len = docs_len, total=total))
    return render_template("activelearning.html", text =text, predictions=labels, docs_len=docs_len, document_id=document_id, total=total ) 

    



### lABELLING THE TOPIC AND SAVING THE RESPONSE aaa
@app.route("//get_label//<document_id>//", methods=["POST", 'GET'])
def get_label(document_id):
    document_id = document_id 
    user_id=session["user_id"] 
    # get_document_information = url + "//get_document_information"
    # data = requests.post(get_document_information, json={ "document_id": document_id,
    #                                                     "user_id":user_id
    #                                                      }).json()

    data = get_doc_info(document_id, user_id)
                                                         
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
        return redirect('/login')



@app.route('//non_active_label//<name>//<document_id>/', methods=["POST", "GET"])
def non_active_label(name, document_id):
    # get_document_information = url + "//get_document_information"
    # response = requests.post(get_document_information, json={ "document_id": document_id,
    #                                                     "user_id":session["user_id"]
    #                                                      }).json()
    response = get_doc_info(document_id, session["user_id"])                                                    

    text = all_texts["text"][str(document_id)]
    words = get_words(response["topic"],  text)
    labels = list(set(session["labels"].strip(",").split(",")))
    # print("start time")
    session["start_time"] = str(session["start_time"]) + "+" + str(datetime.now().strftime("%H:%M:%S"))
    # print(session["start_time"])
    total = len(all_texts["text"].keys())
    docs = list(set(session["labelled_document"].strip(",").split(",")))
    docs_len = len(docs)
    print(docs_len)
    print(response)



    if request.method =="POST":
        label = request.form.get("label").lower()
        drop = request.form.get("suggestion").lower()
        label = str(drop)+str(label)

        name=name 
        document_id=str(document_id)
        user_id = session["user_id"]

        st = datetime.strptime(session["start_time"].strip("+").split("+")[-2], "%H:%M:%S")
        et = datetime.strptime(session["start_time"].strip("+").split("+")[-1], "%H:%M:%S")

        response_time = str(et-st)

        
        
        # recommend_document = "https://nist-topic-model.umiacs.umd.edu/recommend_document"
        # recommend_document = url + "//recommend_document"
        # posts = requests.post(recommend_document, json={
        # "user_id" : int(user_id),
        # "label": label,
        # "response_time": str(response_time),
        # "document_id" : document_id
        # }).json()

        posts = nist_recommend(int(user_id), label, document_id, response_time)
        next = posts["document_id"]
        predictions.append(label.lower())
        
        
        session["labelled_document"] = session["labelled_document"]+","+str(document_id)
        docs = list(set(session["labelled_document"].strip(",").split(",")))
        session["labels"] = session["labels"] + "," + label
        labels = list(set(session["labels"].strip(",").split(",")))
        docs_len = len(docs)
        save_labels(session)

        save_response(name, label, response_time, document_id, user_id)
        # get_document_information = url + "//get_document_information"
        # response = requests.post(get_document_information, json={ "document_id": posts["document_id"],
        #                                                 "user_id":session["user_id"]
        #                                                  }).json()
        
        response = get_doc_info(posts["document_id"], session["user_id"])    
        print(docs_len)
        flash("Response has been submitted")
        
        total = len(all_texts["text"].keys())
        done = len(docs)
        # print('trying to redirect to non active label again')
        return redirect(url_for("non_active_label", name=name, document_id=posts["document_id"]))
        # return redirect(url_for("non_active_label", response=response, words=words, document_id=posts["document_id"], name=name, predictions=labels, pred=response["prediction"], total=total, docs_len=docs_len))

    return render_template("nonactivelabel.html", response=response, words=words, document_id=document_id, text=text, name=name, predictions=labels, pred=response["prediction"], total=total, docs_len=docs_len)

@app.route("/non_active/<name>/<topic_id>//<documents>//<keywords>")
def topic(name, topic_id, documents, keywords): 
    print(topic_id)
    # res = get_single_document(documents, all_texts)
    keywords = keywords.strip("'[]'").split("', '")
    docs = list(set(session["labelled_document"].strip(",").split(",")))
    docs_len = len(docs)
    
    res = get_single_document(documents.strip("'[]'").split(", "), all_texts, docs=docs)

    return  render_template("topic.html", res = res, topic_id=topic_id, docs_len = docs_len, keywords=keywords) 
 
 

@app.route("/<name>/labeled/<document_id>")
def view_labeled(name,document_id):
    text = all_texts["text"][document_id]
    response = extract_label(name, document_id )
    
    return render_template("viewlabeled.html", text=text, response=response, document_id=document_id) 



@app.route("/<name>/labeled_list/")
def labeled_list(name):
    labe = session["labelled_document"]
    
    docss = labelled_docs(labe, all_texts)
    completed = completed_json_(name)
    completed_docs = get_completed(completed, all_texts)
    return render_template("completed.html", completed_docs=completed_docs, docss=docss)
 