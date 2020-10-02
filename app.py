from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("startup_pred.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    for x in request.form.values():
        if x.isnumeric():
            int_features=[int(x) for x in request.form.values()]
            final=[np.array(int_features)]
            print(int_features)
            print(final)
            prediction=model.predict(final)
            np.set_printoptions(precision=0)
            x=prediction[0]
            print(int(x))
            #output='{0:.{1}f}'.format(prediction[0][1], 2)

            if x:
                return render_template('startup_pred.html',pred='Profits that your startup might yield would be around {}'.format(int(x)))
            else:
                return render_template('startup_pred.html',pred='Enter Your Data Properly')


if __name__ == '__main__':
    app.run(debug=True)

