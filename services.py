import os,base64
from datetime import timedelta
from flask import Flask, request, render_template, session, Markup, send_from_directory
from flask import jsonify,send_file,redirect,url_for
from main import gen
from flask_cors import *
from FaceRecognizer import face_train,match

app = Flask(__name__)
app.config['SECRET_KEY']=os.urandom(24)    #设置为24位的字符，每次运行服务器都是不同的，所以服务器启动一次上次的session就清除
app.config['DEBUG']=True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
CORS(app, supports_credentials=True) #跨域


#人脸识别接口
@app.route('/facePost',methods=['POST','GET'])
def facePost():
    #canvas = request.form.get('data')
    img_str = request.values['cs']
    img_data = base64.b64decode(img_str)
    with open('static/temp_img/001.png', 'wb') as f:
        f.write(img_data)
        f.close()
    i,j = match('static/temp_img/001.png')
    print(i)
    print(j)
    if i=="未识别":
        return jsonify({'data':'未识别'})
    if float(j)  < 40:
        return jsonify({'fid': i})
    return jsonify({'data': '未识别'})


#AI创作古诗
@app.route('/writePoemPost',methods=['POST','GET'])
def ajaxPost():
    words = request.values['words']
    c_format = request.values['format']
    label = request.values['label']
    if label=='1':
        setting = {'model_path': 'checkpoints/tang_199.pth', 'pickle_path': 'tang.npz', 'start_words': words,
               'prefix_words': c_format, 'acrostic': True, 'nouse_gpu': True}
    else:
        setting = {'model_path': 'checkpoints/tang_199.pth', 'pickle_path': 'tang.npz', 'start_words': words,
                   'prefix_words': c_format, 'acrostic': False, 'nouse_gpu': True}
    poem = gen(**setting)
    return jsonify({"c_poem":poem})



app.run(host="127.0.0.1",port=5010)