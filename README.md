# Enron Scandal POI Classifier

The study builds an algorithm that identifies which Enron employees may have committed fraud leading to the company’s collapse in 2002.  

In this project, that prediction is whether an employee was a person of interest (POI) in the Enron scandal, and the features were financial data and email records.  There are 14 financial features ranging from salaries and bonuses to various stock options, and the 6 email features include both emails sent and received and whether persons of interest were on the other end of the messages.

- **poi_id.py**: Python script containing main code.  Script contains all exploratory data cleaning as well as different classifiers and parameters tested. When run, the final function in the file creates the 3 .pkl files also present in this repository.
- **my_classifier.pkl**: Optimal classifier found by poi_id.py.
- **my_dataset.pkl**: Dataset used to train classifier.
- **my_feature_list.pkl**: Final features used in classifier.
- **Enron Submission Questions.pdf**: Description and analysis of the process that went into creating poi_id.py.  Includes answers to guideline questions from Udacity machine learning project.
- **Enron Submission Resource List.pdf**: Sources used throughout the process of building classifier.

Submitted April 2017 as part of Udacity's Data Analysis nanodegree program.
