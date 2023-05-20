import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import base64
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt



st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

counter4 = Counter(train['subscribed'])
counter5 = Counter(train['marital'])
counter6 = Counter(train['job'])

target = train['subscribed']
train = train.drop('subscribed', axis=1)

train = train.drop(['ID','job','education',	'marital',	'default','housing','loan',	'contact','month','previous'], axis=1)
train = pd.get_dummies(train)

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=12)

nb = GaussianNB()
lreg = LogisticRegression()
knn5 = KNeighborsClassifier(n_neighbors = 5)
svc_model = SVC(kernel = 'rbf', gamma = 'scale')
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
rfc = RandomForestClassifier(n_estimators=100, random_state=0)

voting_clf = VotingClassifier(
    estimators=[('Naive Bayes', nb), ('Logistic',lreg), ('KNN', knn5),('SVM', svc_model), ('Decision tree', clf), ('Random Forest', rfc)],
    voting='hard')

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_val)

st.title('Term Deposit Prediction')
st.write('Enter the following details to predict whether the customer will subscribe to a term deposit or not.')

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('term.jpg')    




def app():
  
  col1, col2= st.columns((1,1))

  with col1:
    age = st.number_input("Age:", min_value=0, max_value=100, step=1)
    balance = st.number_input("Balance:", min_value=-100000, max_value=100000, step=1)
    day = st.number_input("Day:", min_value=1, max_value=31, step=1)
    duration = st.number_input("Duration (seconds):", min_value=0, max_value=5000, step=1)
    campaign = st.number_input("Campaign:", min_value=1, max_value=50, step=1)
  
  with col2:
    pdays = st.number_input("Pdays:", min_value=-1, max_value=500, step=1)
    poutcome_failure = st.selectbox("poutcome_failure:", ['0', '1'])
    poutcome_success = st.selectbox("poutcome_success:",['0', '1'])
    poutcome_other = st.selectbox("poutcome_other:", ['0', '1'])
    poutcome_unknown = st.selectbox("poutcome_unknown:", ['0', '1'])


  button = st.button("Predict")

  
  counter = Counter(train['age'])

  st.write('Counter plot of Subscription status')

  st.bar_chart(counter4)

# Plot the counter as a bar chart
  st.write('Counter plot of Age')

  st.bar_chart(counter)

  st.write('Counter plot of duration of calls in campaign')

  counter1 = Counter(train['duration'])

# Plot the counter as a bar chart
  st.bar_chart(counter1)



  counter3 = Counter(train['campaign'])

# Plot the counter as a bar chart

  st.write('Counter plot of Campaign')

  st.bar_chart(counter3)

  st.write('Counter plot of Marital status')


  st.bar_chart(counter5)

  st.write('Counter plot of Job')

  st.bar_chart(counter6)


# Predict term deposit subscription when the button is clicked
  if button:
      # Create a dictionary from the user inputs
      # Create a dictionary from the input values
    input_dict = {
'age':age,
'balance':balance ,
'day':day,
'duration':duration ,
'campaign':campaign ,
'pdays':pdays  ,
'poutcome_failure':poutcome_failure,
'poutcome_other':poutcome_other ,
'poutcome_success':poutcome_success ,
'poutcome_unknown':poutcome_unknown}

    # Convert the dictionary into a Pandas DataFrame
    input_df = pd.DataFrame.from_dict([input_dict])

    # Use the pre-trained SVM model to make a prediction
    prediction = voting_clf.predict(input_df)

    # Return the prediction
    

    # Display the prediction
    if prediction[0] == 1:
          st.write("The customer is likely to subscribe to the term deposit.")
    else:
          st.write("The customer is unlikely to subscribe to the term deposit.")


if __name__=='__main__':
    app()

