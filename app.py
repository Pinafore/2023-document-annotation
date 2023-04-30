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
#     doc_id = request.json.get('doc_id')
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

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)



from flask import Flask, request, session, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import json
import random
from backend_server import User

app = Flask(__name__)

# Configure Flask-PyMongo
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mydatabase'
mongo = PyMongo(app)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/create_user', methods=['POST'])
def create_user():
    # mode = random.randint(1, 4)
    mode = 1
    user = User(mode)
    user_id = mongo.db.users.insert_one({"mode": mode}).inserted_id
    session['user_id'] = str(user_id)
    session['user_obj'] = user
    return jsonify({"user_id": str(user_id), "mode": mode})

@app.route('/recommend_document', methods=['POST'])
def nist_recommend():
    user_id = session.get('user_id')
    user = session.get('user_obj')

    if not user_id or not user:
        return jsonify({"code": 400, "msg": "User not found"})

    label = request.json.get('label')
    doc_id = request.json.get('doc_id')
    response_time = request.json.get('response_time')

    result = user.round_trip1(label, doc_id, response_time)
    result['code'] = 200
    result['msg'] = 'SUCCESS'

    print('SUCCESS. Sending result to frontend')
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)