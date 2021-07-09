
import pandas as pd
import pickle
from flask import Flask , render_template,request



app = Flask(__name__)

data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location = request.form['location']
    bhk = request.form['bhk']
    bath = request.form['bath']
    sqft= request.form['total_sqft']

    print(location,bhk,bath,sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]], columns=['location','total_sqft','bath','bhk'])
    predection = pipe.predict(input)[0]*1e5

    return str(predection)






if __name__ == "__main__":
    app.run(debug=True, port=5000)    