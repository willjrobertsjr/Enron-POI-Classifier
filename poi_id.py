#!/usr/bin/python
# -*- coding: cp1252 -*-

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

###List of all features separated by type

###financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
###'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
###'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
###'shared_receipt_with_poi']

###email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
###'shared_receipt_with_poi']

###POI label: [‘poi’]



### Task 1: All given features.

features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

###look at list of POIs 
#for person in data_dict:
#    poi = data_dict[person]["poi"]
#    if poi == 1:
#        print person

###look at length of employee list
#len(data_dict)

###look at the number of features for employees
#len(data_dict['LAY KENNETH L'])

###plot bonus and salary
#for point in data:
#    salary = point[0]
#    bonus = point[1]
#    matplotlib.pyplot.scatter( salary, bonus )

#matplotlib.pyplot.xlabel("bonus")
#matplotlib.pyplot.ylabel("salary")
#matplotlib.pyplot.show()

###look at list for people with large salary and bonus
#for person in data_dict:
#    salary = data_dict[person]["salary"]
#    bonus = data_dict[person]["bonus"]
#    if salary != 'NaN' and bonus != 'NaN' and salary > 1000000 and bonus > 5000000:
#        print person

###TOTAL is not an employee, so need to remove from dataset
data_dict.pop("TOTAL", 0) 

###after seeing TOTAL not a person, look at rest of employee list and see another non-employee
data_dict.pop("The Travel Agency In the Park", 0)

###prepare data with regard to 'NaN's and transform to Dataframe for easier viewing
data_dict_pd = [{k: data_dict[p][k] if data_dict[p][k] != 'NaN' else None for k in data_dict[p].keys()} for p in data_dict.keys()]
data_pd = pd.DataFrame(data_dict_pd)
employees = pd.Series(list(data_dict.keys()))

###add employee names to dataframe and make a series with employee and number of 
###filled values they have. Remember all have at least 1 (POI)
data_pd.set_index(employees, inplace = True)
variable_counts = data_pd.count(axis=1, level=None, numeric_only=True)
variable_counts.sort_values(ascending=True, inplace=True)

#print variable_counts

###one person with no values for anything but poi, not a useful person if nothing is known about them
#print data_dict['LOCKHART EUGENE E']

###remove person from set that will be used to aggregate
data_dict.pop("LOCKHART EUGENE E")

###look at boxplot of salaries with cleaned information now
#ax = data_pd.boxplot('salary', by = 'poi')
###save plot:
#fig = ax.get_figure()
#fig.savefig('temp.png')
#plt.show()



### Task 3: Create new features

###Created variables for percentage of exercised stock to total (exercised_and_totalstocks)
###Percentage of emails to poi compared to all emails sent (from_poi_emails)
###Percentage of emails from poi to person compared to all emails received (to_poi_emails)

###function creates list of employees for created feature. Takes into account NaN values and assign 0 
def create_proportion_variable(subset,total):
    list=[]
    for person in data_dict:
        if data_dict[person][subset]=="NaN" or data_dict[person][total]== "NaN":
            list.append(0)
        elif data_dict[person][subset]>=0:
            list.append(float(data_dict[person][subset])/float(data_dict[person][total]))
    return list

### create three lists of new features using function above
exercised_and_totalstocks = create_proportion_variable("exercised_stock_options","total_stock_value")
from_poi_emails = create_proportion_variable("from_this_person_to_poi","from_messages")
to_poi_emails = create_proportion_variable("from_poi_to_this_person","to_messages")

###add variables into full data_dict employee set
count=0
for person in data_dict:
    data_dict[person]["exercised_and_totalstocks"] = exercised_and_totalstocks[count]
    data_dict[person]["from_poi_emails"] = from_poi_emails[count]
    data_dict[person]["to_poi_emails"] = to_poi_emails[count]
    count = count + 1

#### Store to my_dataset for easy export to pickle below.
my_dataset = data_dict

### Extract features and labels from dataset for testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Tasks 4.  Try a varity of classifiers.
#separate data to train and test
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

###Classifier List
knn = KNeighborsClassifier()
nb = GaussianNB()
tree = DecisionTreeClassifier()

###out of box classifier test
#clf = nb
#clf.fit(features_train,labels_train)
#pred = clf.predict(features_test)

###test classifiers without tuning.  Also used to determine metrics after 3 new features introduced.
#print test_classifier(clf, my_dataset, features_list)




### Task 5.  Tune your classifier to achieve better than .3 precision and .3 recall.
### Parameter List

###Function with PCA, not best classifier for NB or KNN
def test_algorithm_pca(classifier, parameters):
    #create components for pipeline
    scaler = StandardScaler()
    select = SelectKBest(k='all')
    pca = PCA()
    #use feature union to help with n_component and kbest compatibility
    combined_features = FeatureUnion([('feature_selection', select), ('pca', pca)])
    #create list of steps and feed into pipline
    steps = [('scaler', scaler), ('features', combined_features), ('clf', classifier)]
    pipeline = Pipeline(steps)
    #use sss to split data into test and train given small dataset
    sss = StratifiedShuffleSplit(labels, n_iter=100, random_state = 42)
    #use gridsearch to tune algorithm and features for optimal parameters
    cv = GridSearchCV(pipeline, parameters, cv=sss, scoring="f1")
    #fit best parameters for algorithm to dataset
    cv.fit(features, labels)
    clf = cv.best_estimator_
    #pull out steps from feature union to find features used
    union = clf.named_steps['features']
    skb_union = union.get_params()['feature_selection']
    #create list of features used given indices in skb_union
    features_selected = [features_list[i+1] for i in skb_union.get_support(indices=True)]
    #create list of feature scores given indices in skb_union
    features_scores = [skb_union.scores_[i] for i in skb_union.get_support(indices=True)]

    print 'The results for the classifier:', classifier 
    print 'The features selected by SelectKBest:'
    print features_selected
    print features_scores
    print 'Best parameters:', cv.best_params_
    #Task 6: Dump classifier, dataset, and features list to pickle files.
    dump_classifier_and_data(clf, my_dataset, features_list)
    print 'Test Classificatier' 
    test_classifier(clf, my_dataset, features_list)
    return clf

###Final function. Includes same steps as function above without PCA.
###Used to create final pickle files.
def test_algorithm(classifier, parameters):
    scaler = StandardScaler()
    select = SelectKBest()
    steps = [('scaler',scaler), ('feature_selection', select), ('clf', classifier)]
    pipeline = Pipeline(steps)
    sss = StratifiedShuffleSplit(labels, n_iter=100, random_state = 42)
    cv = GridSearchCV(pipeline, parameters, cv=sss, scoring="f1")
    cv.fit(features, labels)
    clf = cv.best_estimator_
    #create list of features used given indices in feature selection step
    features_selected = [features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
    #create list of feature scores matching scores and indices from feature selection step
    features_scores = [clf.named_steps['feature_selection'].scores_[i] for i in clf.named_steps['feature_selection'].get_support(indices=True)]

    print 'The results for the classifier:', classifier 
    print 'The features selected by SelectKBest:'
    print features_selected
    print 'The feature scores for features selected by SelectKBest:'
    print features_scores
    print 'Best parameters:', cv.best_params_
    dump_classifier_and_data(clf, my_dataset, features_list)
    print 'Tester Classification report' 
    test_classifier(clf, my_dataset, features_list)
    return clf

###Parameter tuning and feature selection choices 

###Naive Bayes Testing
#parameters_nb = {'features__feature_selection__k': [1, 5, 10, 15]}
#test_algorithm(nb, parameters_nb)

###2 above and 2 below optimal k from above
#parameters_nb = {'features__feature_selection__k': [3, 4, 5, 6, 7]}
#test_algorithm(nb, parameters_nb)

###Introduce PCA
#parameters_nb_pca = {'features__feature_selection__k': [3, 4, 5, 6, 7], 
                 #'features__pca__n_components': [1,2,3], 
                 #'features__pca__whiten': [True, False]} 
#test_algorithm_pca(nb, parameters_nb_pca)



###KNN Testing
#parameters_knn = {'feature_selection__k': [1, 5, 10, 15],
                  #'clf__n_neighbors' : [2, 6, 10, 15],
                  #'clf__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']}
#test_algorithm(knn, parameters_knn)


###2 above and 2 below default n_neighbors
#parameters_knn = {'feature_selection__k': [3, 4, 5, 6],
                  #'clf__n_neighbors' : [1, 2, 3, 4]
                  #'clf__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']}
#test_algorithm(knn, parameters_knn)


###Given choice of more n_neighbors, 5 seems to optimal.  Same as first CV.
#parameters_knn = {'feature_selection__k': [3, 4, 5, 6],
                  #'clf__n_neighbors' : [5, 6, 7, 10],
                  #'clf__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']}
#test_algorithm(knn, parameters_knn)

###Introduce PCA
#parameters_knn_pca = {'features__feature_selection__k': [3, 4, 5, 6],
                      #'clf__n_neighbors' : [1, 2, 3, 4],
                      #'clf__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                      #'features__pca__n_components': [1,2,3], 
                      #'features__pca__whiten': [True, False]}
#test_algorithm_pca(knn, parameters_knn_pca)



###Decision Tree Testing
#parameters_tree = {'feature_selection__k': [2, 5, 10, 15],
                   #'clf__criterion': ['gini', 'entropy'],
                   #'clf__min_samples_split': [2, 10, 15, 20],
                   #'clf__min_samples_leaf': [1, 5, 10]}
#test_algorithm(tree, parameters_tree)

###More features since 15 was optimal                        
#parameters_tree = {'feature_selection__k': [12, 13, 14, 15, 16, 17],
                   #'clf__criterion': ['gini', 'entropy'],
                   #'clf__min_samples_split': [2, 10, 15, 20],
                   #'clf__min_samples_leaf': [1, 5, 10]}
#test_algorithm(tree, parameters_tree)

###Less features since 12 was optimal                       
#parameters_tree = {'feature_selection__k': [8, 9, 10, 11, 12],
                   #'clf__criterion': ['gini', 'entropy'],
                   #'clf__min_samples_split': [2, 10, 15, 20],
                   #'clf__min_samples_leaf': [1, 5, 10]}
#test_algorithm(tree, parameters_tree)




###FINAL OPTIMAL CLASSIFIER.  Run and create pickles files.
nb = GaussianNB()
parameters_nb = {'features__feature_selection__k': [3, 4, 5, 6, 7]}
test_algorithm(nb, parameters_nb)
