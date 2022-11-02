import numpy as np
from flask import Flask,render_template,jsonify,request
import pickle

app=Flask(__name__)

app =Flask(__name__)
model = pickle.load(open('./models/lr.pkl','rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/models")
def model():
    return render_template('models.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/linear-reg")
def lr():
    return render_template('linear-reg.html')


@app.route('/linearpredict',methods=['POST'])
def linearpredict():
    
    model = pickle.load(open('./models/lr.pkl', 'rb'))
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0], 2)
    return render_template('linear-reg.html', prediction_text='Your salary : {}'.format(output))

if __name__  == "__main__":
    app.run(debug=True)