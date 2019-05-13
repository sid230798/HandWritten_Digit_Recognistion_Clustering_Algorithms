'''

Name :- Siddharth Nahar
Entry No :- 2016csb1043
Date :- 17/11/18
Purpose :-

	1. Driver Code performs calling of function

'''

from PreProcess import preProcess
from scipy.cluster.vq import kmeans, whiten
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#------------------------------------------------------------------------------------

#Form New Clusters by existing centroids
def formClusters(Obs, Labels, Centroids):

	#Initialise All clusters with empty list
	Clusters = list()

	for i in range(Centroids.shape[0]):
		Clusters.append(list())

	for indexLabel, X in enumerate(Obs):

		minDist = float("inf")
		minIndex = 0

		#For each centroid check from which it is at minimum and insert in it
		for index, centroid in enumerate(Centroids):

			dist = np.linalg.norm(X - centroid)
			if dist <= minDist:
				
				minDist = dist
				minIndex = index	

		#Append in that list
		Clusters[minIndex].append((Labels[indexLabel],X))

	return Clusters

#---------------------------------------------------------------------------------------------

def assignLabels(clusters, centroids):

	labelled_centroids = []

	#Iterate through each cluster and corresponding centroid
	for i in range(len(clusters)):
		labels = list(map(lambda x: x[0], clusters[i]))

        	# pick the most common label
		most_common = max(set(labels), key=labels.count)

		#Append it as tuple of label ,centroid row
		centroid = (most_common, centroids[i])
		labelled_centroids.append(centroid)

	#Return the labelled list
	return labelled_centroids

#----------------------------------------------------------------------------------------------

def display(digit, k):

	'''
	image = digit
	plt.figure()
	fig = plt.imshow(image.reshape(20,20))
	fig.set_cmap('gray_r')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)
	if title != "":
		plt.title("Inferred label: " + str(title))

	plt.show()
	'''

	maxPlot = k + 4 - k%4
	rows = maxPlot/4

	plt.rcParams.update({'font.size': 4})
	fig = plt.figure()
	fig.subplots_adjust(hspace = k/50, wspace=0.2)
	for i in range(1, k+1):

		ax = fig.add_subplot(rows, 4, i)
		

		if(i > len(digit)):
			f = ax.text(0.5, 0.5, str(),fontsize=18, ha='center')
			title = ""
		else:	

			img = digit[i-1][1].reshape(20,20)
			img = np.fliplr(img)
			img = np.rot90(img)
			f = ax.imshow(img)
			#f = ax.imshow(digit[i-1][1].reshape(20,20))
			f.set_cmap('gray_r')
			title = "Inferred Label : "+str(digit[i-1][0])
		
		f.axes.get_xaxis().set_visible(False)
		f.axes.get_yaxis().set_visible(False)
		ax.set_title(title)

	plt.savefig("k = "+str(k))
	plt.show()

#------------------------------------------------------------------------------------------------------

def classify_digit(digit, centroids):

	minDist = float("inf")
	predLabel = None
	
	for label, c in centroids:

		dist = np.linalg.norm(digit - c)
		if dist <= minDist:
				
			minDist = dist
			predLabel = label

	return predLabel

#-----------------------------------------------------------------------------------------------------

def getError(Test, labels, Centroids):

	countWrong = 0
	pred = list()

	for index, digit in enumerate(Test):

		predicted = classify_digit(digit, Centroids)
		pred.append(predicted)
		if predicted != labels[index] :

			countWrong += 1

	#print(len(labels),len(pred))
	print(confusion_matrix(labels, pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

	print("Training Accuracy : " +str(1 - countWrong/len(Test)))	

#------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	Obs, labels = preProcess()
	k = 10

	Centroids = kmeans(Obs, k, iter = 50)
	cenArray = np.array(Centroids[0])
	
	#Get Required Clusters
	Clusters = formClusters(Obs, labels, cenArray)
	labelled_Centroids = assignLabels(Clusters, cenArray)

	labelled_Centroids.sort(key = lambda x : x[0])
	'''
	for label, digit in labelled_Centroids:
		display(digit, label)
	'''
	#print(Centroids)
	getError(Obs, labels, labelled_Centroids)
	display(labelled_Centroids, k)
	
