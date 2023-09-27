from flask import Flask, url_for,render_template,request
import pandas as pd
import pickle

loaded_model = pickle.load(open('Random_Forest_regreesor.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('Real_Combine_test.csv')
    my_prediction = loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction = my_prediction.tolist()
    return render_template('result.html',prediction = my_prediction)


if __name__=='__main__':
    app.run(debug=True)