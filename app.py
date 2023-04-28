from flask import Flask, request
import json
from backend_server import Session
app = Flask(__name__)

new_sess = Session(1)
# new_sess.__init__()


@app.route('/')
def hello():
    return 'Hello World!'

# @app.route('/start_model')
# def start():
#     new_sess = Session()
#     result = dict()
#     # Remember to also return first recommended doc id
#     # result = new_sess.get_all_doc()
#     # recommendation = new_sess.highlight_doc()
#     # result['Document_id'] = recommendation['Document_id']
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     result['success'] = True
#     # return n
#     # new_sess.__init__()
#     return result
#     # return {
#     #     "code": 200,
#     #     "data": {"message": "Success"},

#     # }

# @app.route('/getAllDocs')
# def initialize():
#     # res = new_sess.get_all_doc()
#     # print(res)
#     # with open("./Data/random_shit.json", "w") as outfile:
#     #         json.dump(res, outfile)
#     result = new_sess.get_all_doc()
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     result['success'] = True
#     return result

# @app.route('/recommend')
# def recommend():
#     result = new_sess.highlight_doc()
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     result['success'] = True
#     return result

# @app.route('/userClick', methods = ['POST'])
# def click():
#     strip = request.json.get('Document_id')
#     doc_id = int(strip)
#     result = new_sess.get_topics(doc_id)
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     result['success'] = True
#     return result

# @app.route('/recommendByLabel', methods = ['POST'])
# def recommend1():
#     label = int(request.json.get('apply'))
#     doc_id = int(request.json.get('Document_id'))
#     new_label = request.json.get('New_label')

#     result = dict()
#     result['Documend_id'] = 3
#     # return result
#     new_sess.label_doc(label, new_label, doc_id)
#     # return result
#     result = new_sess.highlight_doc()
#     result['code'] = 200
#     result['msg'] = 'SUCCESS'
#     result['success'] = True
#     return result



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
    result['success'] = True
    return result





# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)