from flask import Flask, request
import json
from backend_server import Session
app = Flask(__name__)



new_sess = Session(3)
# new_sess.__init__()


@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/restart_session')
def restart():
    new_sess = Session(2)
    result = {}
    result['code'] = 200
    result['msg'] = 'SUCCESS'
    return result

@app.route('/recommend_document', methods = ['POST'])
def nist_recommend():
    label = request.json.get('label')
    # new_label = request.json.get('new_label')
    doc_id = request.json.get('doc_id')
    response_time = request.json.get('response_time')
    # recommend = request.json.get('recommend')
    # result = {}

    
    result = new_sess.round_trip1(label, doc_id, response_time)

    result['code'] = 200
    result['msg'] = 'SUCCESS'
    
    print('SUCCESS. Sending result to frontend')
    # print(result)
    return result





# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)

# from flask import Flask, session, request, redirect, render_template
# from flask_session import Session

# app = Flask(__name__)

# # Configure Flask-Session
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_PERMANENT'] = False
# Session(app)

# @app.route('/store', methods=['POST'])
# def store_data():
#     session['custom_data'] = request.json.get('data')
#     return redirect('/retrieve')

# @app.route('/retrieve', methods=['GET'])
# def retrieve_data():
#     custom_data = session.get('custom_data', 'No data found.')
#     return f"Custom data: {custom_data}"

# if __name__ == '__main__':
#     app.run(debug=True)