'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 17/11/18
Purpose :-

	1. Stores each data point in list of Vector objects

'''

import numpy as np
import re

def preProcess():

	#Store list of Objects
	DataObservations = list()
	DataLabels = list()

	pattern = re.compile('\d')
	#Read two files simultaneously and get corressponding vectors
	with open("Data/data.txt",'r') as filePtr1, open("Data/label.txt") as filePtr2 :

		for vectorString, labelString in zip(filePtr1, filePtr2): 

			vectorString = vectorString.strip()
			labelString = labelString.strip()

			labelList = pattern.findall(labelString)

			#Convert list of floats to numpy array
			vector = np.fromstring(vectorString, dtype = np.float64, sep = ',')
		
			#Get Label from cooresponding list of 0,1
			label = (labelList.index('1') + 1)%10

			#Get the required Object and append it to list
			#Obj = labelledData(label, vector)
			DataObservations.append(vector)
			DataLabels.append(label)

	Obs = np.array(DataObservations)

	return Obs,DataLabels
