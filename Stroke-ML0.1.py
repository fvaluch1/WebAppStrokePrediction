# WebApp for Stroke Detection
# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
from sklearn.metrics import accuracy_score
import pickle

# Title ans Subtitle
st.write("""
# Stroke Detection with Machine Learning.
Detects the risk of having a Stroke.
""")

# Create and display Image
image = Image.open("stroke.png")

st.image(image, caption="ML", use_column_width=True)

# Get the data
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Set a Subheader
st.subheader("Data Information")

# Table data
st.dataframe(data)

# statistics data
st.write(data.describe())


# Get the feature input


def get_user_input():
    gender = st.sidebar.selectbox("select gender", ('Male','Female','Other')),	
    age	= st.sidebar.slider("age", 0, 90, 40),
    hypertension = st.sidebar.slider("hypertension",0,1,0),
    heart_disease = st.sidebar.slider("heart_disease",0,1,0),
    ever_married = st.sidebar.selectbox("ever married",("yes","no")),
    work_type = st.sidebar.selectbox("select work type",('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')),
    Residence_type = st.sidebar.selectbox("Select Residencetype",('Urban', 'Rural')),
    avg_glucose_level = st.sidebar.slider("avg_glucose_level",0,250,150),
    bmi = st.sidebar.slider("bmi",0,50,30),
    smoking_status = st.sidebar.selectbox("smoking_status",('formerly smoked', 'never smoked', 'smokes','Unknown'))	

    #Store a dict into a variable
    user = {
        "gender":gender,
        "age":age,
        "hypertension":hypertension,
        "heart_disease":heart_disease,
        "ever_married":ever_married,
        "work_type":work_type,
        "Residence_type":Residence_type,
        "avg_glucose_level":avg_glucose_level,
        "bmi":bmi,
        "smoking_status":smoking_status}
    
    # Transform the data into a DF
    features = pd.DataFrame(user,index=[0])
    return features

# Store the user input
user= get_user_input()

#set a subheader
st.subheader("User Input: ")
st.write(user)

# Create and train

# Prepare the data
data['smoking_status'].replace('Unknown', np.nan, inplace=True)
data['bmi'].fillna(data['bmi'].mean(), inplace=True)
data['smoking_status'].fillna(data['smoking_status'].mode()[0], inplace= True)


# Split the data X and y

X = data.drop(["stroke","id"],axis=1)
y = data["stroke"]

encoder = LabelEncoder()

#X = X.apply(lambda col: encoder.fit_transform(col.astype(str)), axis=0, result_type='expand')
#user = user.apply(lambda col: encoder.transform(col.astype(str)), axis=0, result_type='expand')

categorical_features = ['gender',"ever_married","work_type","Residence_type","smoking_status"]

# make an encoder object
encoder = LabelEncoder()

# fit and transform feature x2
for col in categorical_features:
    encoder.fit(pd.concat([X[col], user[col]], axis=0, sort=False))
    X[col] = encoder.transform(X[col])
    user[col] = encoder.transform(user[col])


X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

#X_train = X_train.reshape(-1, 1)
#y_train = y_train.reshape(-1, 1)
#X_test = X_test.reshape(-1, 1)
#y_test = y_test.reshape(-1, 1)

X.columns
print(X_train)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(user.columns)




#RF = RandomForestClassifier(random_state=42)
#RF.fit(X_train,y_train)

load_clf = pickle.load(open('RF_stroke.pkl', 'rb'))



# Show results
st.subheader("Model Accuracy: ")
st.write(str(accuracy_score(y_test,load_clf.predict(X_test))))

# Store results

preds= load_clf.predict(user)

# Set a subheader and display the classification
st.subheader("Classification: 0 is good - 1 is bad ")
st.write(preds)


#pickle.dump(RF, open('RF_stroke.pkl', 'wb'))
