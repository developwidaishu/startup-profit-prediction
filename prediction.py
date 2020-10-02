import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

dataset = pd.read_csv('Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X, y)

#y_pred = regressor.predict(X_test)

pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



#data = pd.read_csv("Forest_fire.csv")
#data = np.array(data)



