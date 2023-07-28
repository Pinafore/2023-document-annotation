from backend_server import User
import sqlite3
import json


all_texts = json.load(open("./Data/newsgroup_sub_500.json"))
true_labels = list(all_texts['label'].values())
true_sub_labels = list(all_texts['sub_labels'].values())
user_instances = {}
DATABASE = './database/06_27_2023_newsgroup_test.db'


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
                         user_purity REAL,
                         user_RI REAL,
                         user_NMI REAL,
                         topic_information TEXT,
                         FOREIGN KEY (user_id) REFERENCES users (id))''')
        conn.commit()


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
        user = User(mode, user_id)  # Create the User() object and store it in the user_instances dictionary
        user_instances[user_id] = User(mode, user_id)

    print('Creating user', user_id)

    
    return {'user_id': user_id, 'code': 200, 'msg': 'User created'}

def nist_recommend(user_id, label, doc_id, response_time):
    user =  user_instances[user_id]
    ltr, lte, purity, RI, NMI, user_purity, user_RI, user_NMI, result = user.round_trip1(label, doc_id, response_time)
    document_topics = user.get_doc_information_to_save(doc_id)
    document_topics = json.dumps(document_topics)

    actual_label = true_labels[int(doc_id)]

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            cursor.execute('''INSERT INTO recommendations
                            (user_id, label, doc_id, response_time, actual_label, local_training_acc, local_testing_acc, purity, RI, NMI, user_purity, user_RI, user_NMI,topic_information)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (user_id, label, doc_id, response_time, actual_label, ltr, lte, purity, RI, NMI, user_purity, user_RI, user_NMI, document_topics))
            conn.commit()

            result['code'] = 200
            result['msg'] = 'SUCCESS'
            # conn.close()
            return result
        else:
            return {"code": 404, "msg": "User not found"}

def get_list(user_id, recommend_action):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if row is not None:
            user =  user_instances[user_id]
            result = user.get_document_topic_list(recommend_action)
            result['code'] = 200
            result['msg'] = 'SUCCESS'
            return result
        else:
            return {"code": 404, "msg": "User not found"}

init_db()

for j in range(0, 20):
    # creating a user to start a session
    user= create_user()
    user_id = user['user_id']

    recommend_id = get_list(user_id, True)['document_id']
    

    for i in range(len(true_labels)):
        recommend_result = nist_recommend(user_id, true_sub_labels[int(recommend_id)], recommend_id, '0')
        recommend_id = recommend_result['document_id']
        