# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

~~~
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Ex01').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df.dtypes
df=df.astype({'A':'int'})
df=df.astype({'B':'float'})
df.dtypes
from sklearn.model_selection import train_test_split
X=df[['A']].values
Y=df[['B']].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=20)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ai_brain = Sequential([
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_train_scaled,y=y_train,epochs=20000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test_scaled=scaler.transform(x_test)
ai_brain.evaluate(x_test_scaled,y_test)
input=[[100]]
input_scaled=scaler.transform(input)
ai_brain.predict(input_scaled)
```
~~~

## Dataset Information

![image](https://user-images.githubusercontent.com/104999433/195591836-a5a6f998-430d-49d4-a3af-3443fc115325.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/104999433/195591982-2ce6b5a7-06b7-4649-b833-a947d392efb6.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/104999433/195592095-80d07442-cd9d-4de9-81b3-4b1429754096.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/104999433/195592213-4690a1ab-e780-4380-b843-ec68c73c4e58.png)

## RESULT
A Basic neural network regression model for the given dataset is developed successfully.
