#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import pandas
import numpy as np
sys.path.append("../tools/")
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### Features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''### SVC Features
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income', 
                 'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                 'scaled_share_with', 'loan_advances', 'expenses']'''

### DecisionTree Features
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income',
                'long_term_incentive', 'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                'scaled_share_with']

'''### Final selected features
features_list = ['poi', 'salary','bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 
                 'long_term_incentive']'''

'''### Original Features (With two removed)                 
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees'
                 , 'to_messages', 'from_poi_to_this_person','from_messages', 'from_this_person_to_poi'
                 , 'shared_receipt_with_poi']'''
                 
'''### Full features list (With new features)
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'long_term_incentive', 
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'scaled_share_with', 'scaled_from_poi']'''

'''### Selected six + two new features               
features_list = ['poi', 'salary','bonus', 'deferred_income', 'total_stock_value', 'exercised_stock_options', 
                 'long_term_incentive', 'scaled_share_with', 'scaled_from_poi']'''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Remove key; TOTAL from data
data_dict.pop('TOTAL', 0)

### Create and add new features into dataset
for k, v in data_dict.items():
    if data_dict[k]["shared_receipt_with_poi"] == "NaN":
        data_dict[k]["scaled_share_with"] = "NaN"
    else:
        data_dict[k]["scaled_share_with"] = float(data_dict[k]["shared_receipt_with_poi"]) 
        / float(data_dict[k]["to_messages"])  
    if data_dict[k]["from_poi_to_this_person"] == "NaN":
        data_dict[k]["scaled_from_poi"] = "NaN"
    else:
        data_dict[k]["scaled_from_poi"] = float(data_dict[k]["from_poi_to_this_person"]) 
        / float(data_dict[k]["to_messages"])

### Function to extract feature valuse from dataset, for plotting purposes
def get_data_from_dict(feature):
    
    """
        Function to extract values of selected features from dataset
    """
    
    data = []
    
    for key, value in data_dict.items():
        data.append(data_dict[key][feature])  
    return data

### Function to check if data has NaN values
def check_nan(feature):
    
    """
        Check if dataset has NaN values
    """
    
    check = False
    for f in feature:
        if f == 'NaN':
            check = True
            break
        else:
            continue
    return check

'''### Creating dataFrame from dictionary - pandas
# Pandas function to retrieve summary statistics for outliers
df = pandas.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
print df.describe().loc[:,['salary', 'bonus', 'scaled_from_poi']]'''

'''### Check if features have NaN values
for x in range(len(features_list)):
    dataset_data = []
    dataset_data = get_data_from_dict(features_list[x])    
    print "Feature: {}" .format(features_list[x]), ", ", "Has Nan: {}" .format(check_nan(dataset_data))'''

'''### Print scatter-plot to check for relations of new features
poi = get_data_from_dict("poi")
scaled_share_with = get_data_from_dict("scaled_share_with")
scaled_from_poi = get_data_from_dict("scaled_from_poi")

plt.scatter(scaled_share_with, scaled_from_poi, c=poi)
plt.title("Scaled Shared Receipt with PoI vs Scaled From PoI to This Person")
plt.xlabel("Scaled Shared Receipt with PoI")
plt.ylabel("Scaled From PoI to This Person")
plt.grid()
plt.show()'''

'''### Total number of data points
print "Total Number of Data Points: {}" .format(len(data_dict))

### Allocation Across Classes
poi_counter = 0
for key, value in data_dict.items():
    if data_dict[key]["poi"] == True:
        poi_counter += 1

print "Number of PoIs: {}" .format(poi_counter)
print "Number of Non-PoIs: {}" .format(len(data_dict) - poi_counter)'''

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
'''### Create SelectPercentile classifier
spercentile = SelectPercentile()'''

'''### SVC and SelectPercentileparameters to test/estimate for
clf_params = {'clf__C': [1e-5, 10, 1e2, 1e5],
             'clf__gamma': [0.0],
             'clf__kernel': ['linear', 'poly', 'rbf'],
             'spercentile__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

### For SVC pipeline, iterate test 50 times
pipe = Pipeline(steps=[('spercentile', spercentile), ('minmaxer', MinMaxScaler()), ('clf', SVC())])
cross_val_ss = StratifiedShuffleSplit(labels,n_iter = 50, random_state = 42)
a_grid_search = GridSearchCV(pipe, param_grid = clf_params, cv = cross_val_ss, scoring = 'recall')
a_grid_search.fit(features,labels)

### Print out best suited parameters and recall score for SVC
print 'SVC Best Parameters: {}' .format(a_grid_search.best_params_)
print 'Best Recall Score: {}' .format(a_grid_search.best_score_)
print 'Best SVC Estimator: {}' .format(a_grid_search.best_estimator_)'''

'''### DecisionTree and SelectPercentile parameters to test/estimate for
clf_params= {'clf__min_samples_split' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             'spercentile__percentile': [10, 20, 30, 40, 50, 60, 70, 80, 100],
             'clf__random_state': [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]}

### For DecisionTree pipeline, iterate test 50 times
pipe = Pipeline(steps=[('spercentile', spercentile), ('minmaxer',MinMaxScaler()), \
                       ('clf', tree.DecisionTreeClassifier())])
cross_val_ss = StratifiedShuffleSplit(labels, n_iter = 50, random_state = 42)
a_grid_search = GridSearchCV(pipe, param_grid = clf_params, cv = cross_val_ss, scoring = 'recall')
a_grid_search.fit(features,labels)

### Print out best suited parameters and recall score for DecisionTree
print 'Decision Tree Best Parameters: {}' .format(a_grid_search.best_params_)
print 'Best Recall Score: {}' .format(a_grid_search.best_score_)
print 'Best DecisionTree Estimator: {}' .format(a_grid_search.best_estimator_)'''

'''### Scale features using MinMaxScaler function
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)'''

'''### Use SelectPercentile to find the best 60% features for SVC
select = SelectPercentile(percentile=60)
select = select.fit(scaled_features, labels)'''

'''### Use SelectPercentile to find the best 50% features for DecisionTree
select = SelectPercentile(percentile=50)
select = select.fit(scaled_features, labels)

### Print the features and their rankings
for x in range (len(features_list)-1):
    print "Feature: {}" .format(features_list[x + 1]), ",", "Support: {}" .format(select.get_support()[x]), ",", \
    "Score: {}" .format(select.scores_[x])'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Create list to store results
score_all = []
precision_all = []
recall_all = []

### Set StratifiedShuffleSplit cross validation iteration to 1000 times
folds = 1000

cross_val_ss = StratifiedShuffleSplit(labels, folds, random_state = 42)
for train_indices, test_indices in cross_val_ss:
    ### make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    
    ### Set classifier parameters based on pipeline results
    clf = tree.DecisionTreeClassifier(min_samples_split = 1, random_state = 34)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    score_all.append(clf.score(features_test, labels_test))
    precision_all.append(precision_score(labels_test, pred))
    recall_all.append(recall_score(labels_test, pred))

### Print results of StratifiedShuffleSplit
print 'Precision StratifiedShuffleSplit: {}' .format(np.average(precision_all))
print 'Recall StratifiedShuffleSplit: {}' .format(np.average(recall_all)) 
print 'Accuracy StratifiedShuffleSplit: {}' .format(np.average(score_all)) 
print 'Predictions: {}' .format(pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)