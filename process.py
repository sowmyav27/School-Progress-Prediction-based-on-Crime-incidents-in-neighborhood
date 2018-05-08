from flask import Flask, render_template, request, jsonify
# from withfucntions_255 import startinghere
import pandas as pd
import numpy as np
from math import cos, sqrt
import csv
app = Flask(__name__)

@app.route('/', methods= ["GET", "POST"])
def homepage():

    return render_template('index.html')

@app.route('/randomforrest',methods=["POST"])
def randomforrest():
	result = request.form
	schoolID = int(request.form['school'])
	train_df1 = pd.read_csv('Chicago_Public_Schools_-_School_Progress_Reports_SY1516.csv', sep=',',
							usecols=['School_ID', 'Latitude', 'Longitude',
									 'Student_Growth_Rating',
									 'Teacher_Attendance_Year_1_Pct',
									 'Student_Attendance_Year_1_Pct',
									 'School_Survey_Teacher_Response_Rate_Pct',
									 'One_Year_Dropout_Rate_Avg_Pct',
									 'School_Survey_Student_Response_Rate_Pct',
									 'Suspensions_Per_100_Students_Avg_Pct',
									 'Misconducts_To_Suspensions_Avg_Pct'])
    

	train_df1.fillna(0, inplace=True)
	train_schools = train_df1.iloc[:, :]

	# In[33]:

	train_df2 = pd.read_csv('Chicago_Crime_Data_2015_2016.csv', sep=',',
							usecols=['ID', 'Primary_Type', 'Longitude', 'Latitude'])
	train_df2.fillna(0, inplace=True)
	train_crimes = train_df2.iloc[:, :]

	# In[20]:
	f = open('train_toClassifier.csv', 'w')
	writer = csv.writer(f)
	writer.writerow(
		['Primary_Type', 'Student_Growth_Rating', 'Teacher_Attendance_Year_1_Pct', 'Student_Attendance_Year_1_Pct',
		 'School_Survey_Teacher_Response_Rate_Pct',
		 'One_Year_Dropout_Rate_Avg_Pct', 'School_Survey_Student_Response_Rate_Pct',
		 'Suspensions_Per_100_Students_Avg_Pct',
		 'Misconducts_To_Suspensions_Avg_Pct', 'Distance'])
	deglen = 110.25
	# train_distances = {}
	var1 = 0
	for i in range(len(train_schools)):
		if train_schools['School_ID'][i] == schoolID:
			lat1 = train_schools['Latitude'][i]
			lon1 = train_schools['Longitude'][i]
			# train_distances[train_schools['School_ID'][i]] = {}
			for j in range(len(train_crimes)):
				train_distance = 0
				lat2 = train_crimes['Latitude'][j]
				lon2 = train_crimes['Longitude'][j]
				x = lat2 - lat1
				y = (lon2 - lon1) * cos(lat1)
				train_distance = deglen * (sqrt(pow(x, 2) + pow(y, 2)))
				# print(train_distance)
				# print train_distance
				# row = []

				if train_distance < 3:

					if train_schools['Student_Growth_Rating'][i] == 'ABOVE AVERAGE':
						var1 = 1
					elif train_schools['Student_Growth_Rating'][i] == 'AVERAGE':
						var1 = 2
					elif train_schools['Student_Growth_Rating'][i] == 'BELOW AVERAGE':
						var1 = 3

					writer.writerow(
						[train_crimes['Primary_Type'][j], var1, train_schools['Teacher_Attendance_Year_1_Pct'][i],
						 train_schools['Student_Attendance_Year_1_Pct'][i],
						 train_schools['School_Survey_Teacher_Response_Rate_Pct'][i],
						 train_schools['One_Year_Dropout_Rate_Avg_Pct'][i],
						 train_schools['School_Survey_Student_Response_Rate_Pct'][i],
						 train_schools['Suspensions_Per_100_Students_Avg_Pct'][i],
						 train_schools['Misconducts_To_Suspensions_Avg_Pct'][i],
						 train_distance])
			# print(train_crimes['Primary_Type'][j], train_distance)
			break  # coding: utf-8]

	f.close()
	# In[32]:

	test_df3 = pd.read_csv('train_toClassifier.csv')
	test_df_type = pd.get_dummies(test_df3['Primary_Type'])
	test_df_new = pd.concat([test_df3, test_df_type], axis=1)
	y_test = test_df_new.iloc[:, 1]  # student growth rating
	X_test = test_df_new.iloc[:, 2:10]

	from sklearn.preprocessing import Normalizer
	train_normalizer = Normalizer()
	X_test = train_normalizer.fit_transform(X_test)

	X_train = pd.read_csv('X_train.csv')
	y_train = pd.read_csv('y_train.csv')



   ## Predictions using Random Forest classifier
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators=2, criterion='entropy')
   
    
   ## Predictions using XGB classifier
	import xgboost as xgb
	from xgboost import XGBClassifier
	classifier = XGBClassifier(n_estimators=10,learning_rate=0.3,max_depth=8,
	                           min_child_weight=2,objective= 'binary:logistic')
	
   ## Predictions using SGD classifier
   from sklearn.linear_model import SGDClassifier
   classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
   
	
   classifier.fit(X_train, y_train.values.ravel())
    
   

	y_pred = classifier.predict(X_test)
	print(len(y_pred))
	y_pred = int(np.mean(np.array(y_pred)))

	if y_pred == 1:
		y_pred = 'Above Average'
	elif y_pred == 2:
		y_pred = 'Average'
	elif y_pred == 3:
		y_pred = 'Below Average'

	return render_template("results.html", result=y_pred)



if __name__ == '__main__':
   app.run(host='localhost', port=9999, debug = True)