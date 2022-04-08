"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
#pip install matplotlib
# Streamlit dependencies
import streamlit as st
import joblib,os
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
#import warnings
#warnings.filterwarnings("ignore")




# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

#Load pickled models

#Support Vector Classifier
pickle_in = open("resources/SVC.pkl", "rb")
clf_lsvc = pickle.load(pickle_in)

#Logistic Regression
#pickle_in1 = open('resources/Logistic_regression.pkl', 'rb')
#clf_lr = pickle.load(pickle_in1)

#GnB
pickle_in2 = open("resources/gnb.pkl", "rb")
clf_gnb = pickle.load(pickle_in2)


#Functions to predict the tweet sentiments using the 3 tested classifiers
def predict_tweet_lsvc(tweet):
	"""
	This function makes actual predictions 
	for the support vector classifier

	"""
	prediction_lsvc = clf_lsvc.predict([tweet])
	return prediction_lsvc

#def predict_tweet_lr(tweet):
	"""
	This function makes actual predictions 
	for the logistic regression classifier

	"""
	#prediction_lr = clf_lr.predict([tweet])
	#return prediction_lr

def predict_tweet_gnb(tweet):
	"""
	This function makes actual predictions 
	for the gaussian bernoulli classifier

	"""
	prediction_gnb = clf_gnb.predict([tweet])
	return prediction_gnb

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title(" DANN Analytics Tweet Classifer")
	st.subheader("Climate change tweet classification")
	html_temp = """
	    <div style="background-color:tomato;padding:10px">
		h2 style="color:white;text-align:center;">DANN Analytics Classifier
		<div>
	    """
	st.markdown(html_temp, unsafe_allow_html=True)
	result = ""
    

	# Creating sidebar with selection box -
	# you can create multiple pages this way
options = ["Linear Support Vector Classifier", "Logistic Regression", "Gaussian Bernoulli", "Prediction", "Information", "About DANN Analytics"]
selection = st.sidebar.selectbox("Choose Classifier", options)

visuals = ["Bar chart"]
selection2 = st.sidebar.selectbox("click on the visual", visuals)
#Building out the visuals page
if selection2 == "Bar chart":
	if st.checkbox('Show bar chart'):
		#st.write(train)
		x = train['sentiment'].value_counts()
		style.use('seaborn-pastel')
		sns.set_style('darkgrid')
		fig = plt.figure(figsize=(7,5))
		sns.countplot(x='sentiment', data=train)
		#st.write(fig)
		plt.title("Class Distribution")
		st.pyplot(fig)


	# Building out the "Information" page
if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the "About DANN Analytics" page
if selection == "About DANN Analytics":
		st.info("What we do")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating Filter for models you can choose from
		model_list = ["Linear Support Vector Classifier", "Logistic Regression", "Gaussian Bernoulli"]
		selections = st.selectbox("Choose Classifier", model_list)
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if selections =="Linear Support Vector Classifier":

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success("Text Categorized as: Neutral")
				elif prediction == 1:
					st.success("Text Categorized as: PR0 Climate")
				elif prediction == 2:
					st.success("Text Categorized as: Post News and Facts On Climate")
				elif prediction == -1:
					st.success("Text Categorized as: Post News and Facts On Climate")

		if selections =="Logistic Regression":

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success("Text Categorized as: Neutral")
				elif prediction == 1:
					st.success("Text Categorized as: PR0 Climate")
				elif prediction == 2:
					st.success("Text Categorized as: Post News and Facts On Climate")
				elif prediction == -1:
					st.success("Text Categorized as: Post News and Facts On Climate")

		if selections =="Logistic Regression":

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == 0:
					st.success("Text Categorized as: Neutral")
				elif prediction == 1:
					st.success("Text Categorized as: Pro Climate")
				elif prediction == 2:
					st.success("Text Categorized as: Post News On Climate")
				elif prediction == -1:
					st.success("Text Categorized as: Anti Climate")

		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()