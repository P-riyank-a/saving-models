# prediction 
import pickle

# load model
model=pickle.load(open("diabetic.pkl",'rb'))
data=model.predict([[3,5,0,8,5,4,3,9]])
if data[0]==0:
    print("person is not diabetic")
else:
    print("person is diabetic")  