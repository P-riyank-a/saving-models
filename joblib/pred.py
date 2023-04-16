import joblib

# load model
model=joblib.load(open("diabetic.pkl",'rb'))
data=model.predict([[0,0,0,0,0,1,1,1]])
if data[0]==0:
    print("person is not diabetic")
else:
    print("person is diabetic")  