# from flask import Flask, request
# import json
# from backend_server import User
# app = Flask(__name__)



# new_sess = User(1)
# # new_sess.__init__()



# @app.route('/recommend_document', methods = ['POST'])
# def nist_recommend():
#     label = request.json.get('label')
#     # new_label = request.json.get('new_label')
#     doc_id = request.json.get('document_id')
#     response_time = request.json.get('response_time')
#     # recommend = request.json.get('recommend')
#     # result = {}

    
#     result = new_sess.round_trip1(label, doc_id, response_time)

#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
    
#     print('SUCCESS. Sending result to frontend')
#     # print(result)
#     return result


# @app.route('/restart_session')
# def restart():
#     new_sess = User(2)
#     result = {}
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     return result

# @app.route('/')
# def hello():
#     return 'Hello World!'







from flask import Flask, request, jsonify
from backend_server import User
import random
import sqlite3





app = Flask(__name__)

DATABASE = 'local_users.db'
user_instances = {}
MODES = [0, 1, 2, 3]
MODES = [1, 1, 1, 1]
# MODES = [3, 3, 3, 3]
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

@app.route('/create_user', methods=['POST'])
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
    return {'user_id': user_id, 'code': 200, 'msg': 'User created'}

@app.route('/recommend_document', methods=['POST'])
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



# if __name__ == '__main__':
#     init_db()
#     app.run(host='0.0.0.0', port=3000)
