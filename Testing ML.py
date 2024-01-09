from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = r"C:\Users\adunton\OneDrive - Facebook\Desktop - Andrew Desktop PC\Andrew\Data245\Project\All Lending Club Data (Not from paper)\accepted_2007_to_2018Q4"
training_data_file_name = rf'{path}\accepted_2007_to_2018Q4 post-SMOTE training data.csv'
test_data_file_name = rf'{path}\accepted_2007_to_2018Q4 post-SMOTE test data.csv'
loan_data_file_name = rf'{path}\accepted_2007_to_2018Q4 cleaned encoded.csv'

# training_size = 1000
# test_size = 100

# training_df = pd.read_csv(training_data_file_name, nrows=training_size)
# test_df = pd.read_csv(test_data_file_name, nrows=test_size)
# loan_data_df = pd.read_csv(loan_data_file_name, nrows=training_size)
training_df = pd.read_csv(training_data_file_name)
test_df = pd.read_csv(test_data_file_name)
loan_data_df = pd.read_csv(loan_data_file_name)

#Encoding 2 loan_status values were reversed
training_df['loan_status'] = training_df['loan_status'].map({0: 1, 1: 0})
test_df['loan_status'] = test_df['loan_status'].map({0: 1, 1: 0})
loan_data_df['loan_status'] = loan_data_df['loan_status'].map({0: 1, 1: 0})
training_df = training_df.dropna()
test_df = test_df.dropna()
loan_data_df = loan_data_df.dropna()

# Select the columns for machine learning
selected_columns_ML = [
    "purpose_car",
    "purpose_credit_card",
    "purpose_debt_consolidation",
    "purpose_house",
    "purpose_other",
    "loan_amnt_normalized",
    "term_binary",
    "days_since_issue_d_normalized",
    "installment_normalized",
    "int_rate_normalized",
    "grade_numerical",
    "sub_grade_numerical",
    "tot_cur_bal_normalized",
    "pymnt_plan_binary",
    "addr_state_encoded",
    "total_rec_int_normalized",
    "total_rec_late_fee_normalized",
    "total_rec_prncp_normalized",
    "out_prncp_normalized",
    "emp_title_encoded",
    "emp_length_numerical",
    "annual_inc_normalized",
    "ownership_MORTGAGE",
    "ownership_NONE",
    "ownership_OTHER",
    "ownership_OWN",
    "ownership_RENT",
    "delinq_2yrs_numerical",
    "inq_last_12m_numerical",
    "last_fico_range_low_numerical",
    "mort_acc_numerical",
    "mths_since_last_delinq_numerical",
    "num_actv_bc_tl_numerical",
    "open_il_12m_numerical",
    "tot_hi_cred_lim_normalized",
    "total_acc_numerical",
    "days_since_last_cred_pull_normalized",
    "loan_status_binary"
]

######################################################################################################################
#Define the pre-SMOTE and post-SMOTE training & test datasets

# Filter the dataset to include only the selected columns
#loan_data_df = loan_data_df[selected_columns_ML]

# pre_SMOTE_features = loan_data_df.drop('loan_status_binary', axis=1)
# pre_SMOTE_target = loan_data_df['loan_status_binary']

# pre_SMOTE_features_train, pre_SMOTE_features_test, pre_SMOTE_target_train, pre_SMOTE_target_test = (
#                                             train_test_split(pre_SMOTE_features, pre_SMOTE_target, test_size = 0.2, random_state=42)
#                                             )

# post_SMOTE_features_train = training_df.drop('loan_status_binary', axis=1)
# post_SMOTE_target_train = training_df['loan_status_binary']


# post_SMOTE_features_test = test_df.drop('loan_status_binary', axis=1)
# post_SMOTE_target_test = test_df['loan_status_binary']

pre_SMOTE_features = loan_data_df.drop('loan_status', axis=1)
pre_SMOTE_target = loan_data_df['loan_status']

pre_SMOTE_features_train, pre_SMOTE_features_test, pre_SMOTE_target_train, pre_SMOTE_target_test = (
                                            train_test_split(pre_SMOTE_features, pre_SMOTE_target, test_size = 0.2, random_state=42)
                                            )

post_SMOTE_features_train = training_df.drop('loan_status', axis=1)
post_SMOTE_target_train = training_df['loan_status']


post_SMOTE_features_test = test_df.drop('loan_status', axis=1)
post_SMOTE_target_test = test_df['loan_status']




######################################################################################################################
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize an empty DataFrame to store the performance metrics
performance_metrics = pd.DataFrame(columns=['Model', 'Data Balancing','Notes', 'Class', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Define a function to calculate metrics and add a row to the DataFrame
def add_metrics(performance_metrics, model_name, data_balancing, notes, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'Model': model_name,
        'Data Balancing': data_balancing,
        'Notes': notes,
        'Accuracy': accuracy,
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1_score, support)):
        new_row = pd.DataFrame({
            **metrics,
            'Class': i,
            'Precision': p,
            'Recall': r,
            'F1 Score': f,
            'Support': s
        }, index=[0])
        performance_metrics = pd.concat([performance_metrics, new_row], ignore_index=True)
    return performance_metrics

#####################################################################################################################
# #Logistic Regression. Train both pre-SMOTE and post-SMOTE dataset models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Fit the model on the pre-SMOTE training data
logreg_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions with pre-SMOTE model
target_pred = logreg_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Logistic Regression', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Fit the model on the post-SMOTE training data
logreg_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions with post-SMOTE model
target_pred = logreg_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Logistic Regression', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

# Print the DataFrame
print(performance_metrics)

#####################################################################################################################
# SVM
from sklearn.svm import SVC, LinearSVC

# Initialize the SVM classifier. Use LinearSVC rather than SVC for large sample size
svm_model = LinearSVC()

# Fit the model on the pre-SMOTE training data
svm_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = svm_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'SVM', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the SVM classifier
svm_model = LinearSVC()

# Fit the model on the post-SMOTE training data
svm_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = svm_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'SVM', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

# Print the DataFrame
print(performance_metrics)

#####################################################################################################################
# KNN
from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN classifier
knn_model = KNeighborsClassifier()

# Fit the model on the training data 
knn_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions, use numpy array (error when using dataframe)
features_test_np = pre_SMOTE_features_test.to_numpy()
features_test_np = features_test_np[0:,:]
target_pred = knn_model.predict(features_test_np)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'KNN', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the KNN classifier
knn_model = KNeighborsClassifier()

# Fit the model on the training data 
knn_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions, use numpy array (error when using dataframe)
features_test_np = post_SMOTE_features_test.to_numpy()
features_test_np = features_test_np[0:,:]
target_pred = knn_model.predict(features_test_np)

# Make predictions
target_pred = knn_model.predict(features_test_np)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'KNN', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

# Print the DataFrame
print(performance_metrics)

######################################################################################################################
#Random Forest
from sklearn.ensemble import RandomForestClassifier

# Initialize the random forest classifier
rf_model = RandomForestClassifier()

# Fit the model on the training data 
rf_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the random forest classifier
rf_model = RandomForestClassifier()

# Fit the model on the training data 
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# Stacking XGBoost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier

# Initialize the base models (Random Forest and XGBoost)
rf_model = RandomForestClassifier()

# Initialize the StackingClassifier with Random Forest and XGBoost as base models
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model)],
    final_estimator=XGBClassifier(),
    passthrough=True
)

# Fit the model on the training data
stacking_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(pre_SMOTE_features_test)

# # Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking XGBoost', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the base models (Random Forest and XGBoost)
rf_model = RandomForestClassifier()

# Initialize the StackingClassifier with Random Forest and XGBoost as base models
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model)],
    final_estimator=XGBClassifier(),
    passthrough=True
)

# Fit the model on the training data
stacking_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(post_SMOTE_features_test)

# # Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking XGBoost', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# Stacking AdaBoost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier

# Initialize the base model (Random Forest)
base_rf_model = RandomForestClassifier()

# Initialize the StackingClassifier with Random Forest as the base model and AdaBoost as the meta-learner
stacking_model = StackingClassifier(
    estimators=[('rf', base_rf_model)],
    final_estimator=AdaBoostClassifier(),
    passthrough=True
)

# Fit the model on the training data
stacking_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking AdaBoost', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the base model (Random Forest)
base_rf_model = RandomForestClassifier()

# Initialize the StackingClassifier with Random Forest as the base model and AdaBoost as the meta-learner
stacking_model = StackingClassifier(
    estimators=[('rf', base_rf_model)],
    final_estimator=AdaBoostClassifier(),
    passthrough=True
)
# Fit the model on the training data
stacking_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking AdaBoost', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# Stacking LightGBM (with Random Forest as Base Model)
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier

# Initialize the base model (Random Forest)
base_rf_model = RandomForestClassifier()

# Initialize the ensemble model (LightGBM)
lgbm_model = LGBMClassifier()

# Initialize the StackingClassifier with Random Forest as the base model and LightGBM as the meta-learner
stacking_model = StackingClassifier(
    estimators=[('rf', base_rf_model)],
    final_estimator=lgbm_model,
    passthrough=True
)

# Fit the model on the training data
stacking_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking LightGBM', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the base model (Random Forest)
base_rf_model = RandomForestClassifier()

# Initialize the ensemble model (LightGBM)
lgbm_model = LGBMClassifier()

# Initialize the StackingClassifier with Random Forest as the base model and LightGBM as the meta-learner
stacking_model = StackingClassifier(
    estimators=[('rf', base_rf_model)],
    final_estimator=lgbm_model,
    passthrough=True
)

# Fit the model on the training data
stacking_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking LightGBM', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# XGBoost

# Initialize the XGBoost classifier
xgb_model = XGBClassifier()

# Fit the model on the training data
xgb_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = xgb_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'XGBoost', 'Pre-SMOTE', 'None', pre_SMOTE_target_test, target_pred)

# Initialize the XGBoost classifier
xgb_model = XGBClassifier()

# Fit the model on the training data
xgb_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = xgb_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'XGBoost', 'Post-SMOTE', 'None', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# Random Forest Model With Hyperparameter Tuning

# Creating an instance of Random Forest Classifier with the specified hyperparameters
rf_model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1,
                               max_features='sqrt', max_depth=20)

# Fit the model on the training data
rf_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Pre-SMOTE', 'Hyperparameters', pre_SMOTE_target_test, target_pred)

# Creating an instance of Random Forest Classifier with the specified hyperparameters
rf_model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=1,
                               max_features='sqrt', max_depth=20)

# Fit the model on the training data
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Post-SMOTE', 'Hyperparameters', post_SMOTE_target_test, target_pred)

print(performance_metrics)

# # ######################################################################################################################
# Random Forest Model With Feature Engineering Using Feature Importance Ranking
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Fit the model on the post-SMOTE training data
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)
# Get feature importances
importances = rf_model.feature_importances_
# Get the indices of the features sorted by importance
indices = np.argsort(importances)[::-1]
# Get the feature names
feature_names = post_SMOTE_features_train.columns
# Create a DataFrame of the feature importances
importances_df = pd.DataFrame({
    'Feature': feature_names[indices[:15]],
    'Importance': importances[indices[:15]]
})
# Sort the DataFrame by importance
importances_df = importances_df.sort_values(by='Importance', ascending=True)
# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance Ranking')
plt.tight_layout()
# Save the figure
plt.savefig(rf'{path}\feature_importance.png')
# Initialize lists to store the accuracies and number of features
accuracies = []
num_features = []
# Loop over the features from 5 to the total number of features
for i in range(5, len(feature_names) + 1):
    # Select the top i features
    top_features = [feature_names[j] for j in indices[:i]]
    
    # Select only the top i features for the training data
    selected_features_train = post_SMOTE_features_train[top_features]
    
    # Do the same for the test data
    selected_features_test = post_SMOTE_features_test[top_features]
    
    # Fit the model on the training data
    rf_model.fit(selected_features_train, post_SMOTE_target_train)
    
    # Make predictions
    target_pred = rf_model.predict(selected_features_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(post_SMOTE_target_test, target_pred)
    
    # Add the accuracy and number of features to the lists
    accuracies.append(accuracy)
    num_features.append(i)
# Plot the accuracy over the number of features
plt.figure(figsize=(10, 6))
plt.plot(num_features, accuracies, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy over Number of Features')
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(rf'{path}\accuracy_over_features.png')
# Find the number of features that gives the highest accuracy
max_accuracy_index = accuracies.index(max(accuracies))
best_num_features = num_features[max_accuracy_index]
# Select the top features that give the highest accuracy
best_features = [feature_names[i] for i in indices[:best_num_features]]
# Select only the best features for the training data
best_features_train = post_SMOTE_features_train[best_features]
# Do the same for the test data
best_features_test = post_SMOTE_features_test[best_features]
# Fit the model on the training data with the best features
rf_model.fit(best_features_train, post_SMOTE_target_train)
# Make predictions
target_pred = rf_model.predict(best_features_test)
# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Post-SMOTE', f'{best_num_features} features', post_SMOTE_target_test, target_pred)
print(performance_metrics)

# ######################################################################################################################
# Stacking XGBoost with Random Forest as base model after feature engineering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Fit the model on the post-SMOTE training data
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)
# Get feature importances
importances = rf_model.feature_importances_
# Get the indices of the features sorted by importance
indices = np.argsort(importances)[::-1]
# Get the feature names
feature_names = post_SMOTE_features_train.columns


# Initialize the base models (Random Forest and XGBoost)
rf_model = RandomForestClassifier(random_state=42)

# Initialize the StackingClassifier with Random Forest and XGBoost as base models
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model)],
    final_estimator=XGBClassifier(),
    passthrough=True
)

#best_num_features = 11 #Random Forest with 11 features gave the highest accuracy
best_num_features = 10 #Random Forest with 10 features gave the highest accuracy with outliers removed data
# Select the top features that give the highest accuracy
best_features = [feature_names[i] for i in indices[:best_num_features]]
# Select only the best features for the training data
best_features_train = post_SMOTE_features_train[best_features]
# Do the same for the test data
best_features_test = post_SMOTE_features_test[best_features]

# Fit the model on the training data
stacking_model.fit(best_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(best_features_test)

# # Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking XGBoost', 'Post-SMOTE',
                                   f'{best_num_features} features', post_SMOTE_target_test, target_pred)

print(performance_metrics)

######################################################################################################################
# Hyperparameter tuning with RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Assuming X_train and y_train are your training data and labels

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],     # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt']   # Number of features to consider at every split
}

# Initialize a RandomForestClassifier
rf = RandomForestClassifier()

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# The best hyperparameters from the grid search
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

#Train & test RandomForestClassifier with the best parameters

# Initialize the random forest classifier
rf_model = RandomForestClassifier(
                                n_estimators=best_params["n_estimators"],
                                max_depth=best_params["max_depth"],
                                min_samples_split=best_params["min_samples_split"],
                                min_samples_leaf=best_params["min_samples_leaf"],
                                max_features=best_params["max_features"]
                                )

# Fit the model on the training data 
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(post_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Post-SMOTE', 'Parameter tuning', post_SMOTE_target_test, target_pred)

print(performance_metrics)

######################################################################################################################
#Random Forest with RamdomSearch CV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.ensemble import RandomForestClassifier


# Define the parameter distributions to sample from
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 3, 5, 10],
    'max_features': ['sqrt'],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}
# Create the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42
)

# Fit the model to the data
rf_model.fit(pre_SMOTE_features_train, pre_SMOTE_target_train)

# Make predictions
target_pred = rf_model.predict(pre_SMOTE_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Pre-SMOTE', 'Hyperparameters', pre_SMOTE_target_test, target_pred)

# Print the best hyperparameters found
#print("Best Hyperparameters:", random_search.best_params_)

# Get the best model
#best_rf_model = random_search.best_estimator_

# Make predictions
#y_pred = best_rf_model.predict(X_test)

# Fit the model on the post-SMOTE training data
random_search.fit(post_SMOTE_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = random_search.predict(post_SMOTE_features_test)


# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Random Forest', 'Post-SMOTE', 'Hyperparameters',post_SMOTE_target_test, target_pred)

print(performance_metrics)

# ######################################################################################################################
# Stacking XGBoost with Random Forest as base model after feature engineering & RamdomSearch CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Fit the model on the post-SMOTE training data
rf_model.fit(post_SMOTE_features_train, post_SMOTE_target_train)
# Get feature importances
importances = rf_model.feature_importances_
# Get the indices of the features sorted by importance
indices = np.argsort(importances)[::-1]
# Get the feature names
feature_names = post_SMOTE_features_train.columns

best_num_features = 11 #Random Forest with 11 features gave the highest accuracy
# Select the top features that give the highest accuracy
best_features = [feature_names[i] for i in indices[:best_num_features]]
# Select only the best features for the training data
best_features_train = post_SMOTE_features_train[best_features]
# Do the same for the test data
best_features_test = post_SMOTE_features_test[best_features]

# Define the parameter distributions to sample from
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 3, 5, 10],
    'max_features': ['sqrt'],
    'min_samples_split': [2,5,10],
    'min_samples_leaf': [1,2,4]
}
# Create the Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42
)

# Initialize the StackingClassifier with Random Forest and XGBoost as base models
stacking_model = StackingClassifier(
    estimators=[('rf', rf_model)],
    final_estimator=XGBClassifier(),
    passthrough=True
)

# Fit the model to the data
stacking_model.fit(best_features_train, post_SMOTE_target_train)

# Make predictions
target_pred = stacking_model.predict(best_features_test)

# Add metrics to the DataFrame
performance_metrics = add_metrics(performance_metrics, 'Stacking XGBoost', 'Post-SMOTE', 'Hyperparameters, 11 features', post_SMOTE_target_test, target_pred)

print(performance_metrics)



# ######################################################################################################################
#Save the performance_metrics df as a csv
performance_metrics.to_csv(rf'{path}\ML performance metrics.csv', index=False)

