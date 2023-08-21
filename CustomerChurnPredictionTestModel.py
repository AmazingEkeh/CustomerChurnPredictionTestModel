from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
import seaborn as sns

# Load the dataset for this project
data = pd.read_csv('/Users/user/Desktop/CS677 Data Science with Python/Final Project/Bank Customer Churn Prediction.csv')
print(data)
print(data.info())
print(data.describe())
print()

# Check for missing values
mis = data.isna().sum()
print(mis)
print()

# Convert  categorical variables into numerical representations
data = pd.get_dummies(data, dtype=int)
print(data)
print()

# Checking the proportion of data for the response variable "churn"
churn_counts = data['churn'].value_counts()
print(churn_counts)
print()

# plotting correlation heatmap
cor_plot = data.corr()
sns.heatmap(cor_plot, cmap="Blues", annot=True)
# displaying heatmap
plt.show() # Please view the plot with a full screen as includes figures.

# Features and Labels
X = data.drop(columns=["churn"])
y = data["churn"]

# Splitting the data into training and testing set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, test_size=0.5, random_state=0)

# Perform hyperparameter tuning
# Define the hyperparameter distribution
# Create the random grid
random_grid = {'n_estimator': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': ['auto', 'sqrt']+ [None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [2, 5, 10],
               'bootstrap': [True, False]}

print(f"Random Grid = {random_grid}")
print()


# TRAIN MODELS
# Random Forest Model
rnd_model = RandomForestClassifier()
# Logistic Regression Model
log_model = LogisticRegression()
# Gradiant Boosting Model
gb_model = GradientBoostingClassifier()
# Support Vector Model
svm_model = SVC(probability=True)


# Four classifiers are defined, and a voting classifier is created to combine their predictions.
voting_clf = VotingClassifier(
    estimators = [('rnd', rnd_model), ('lr', log_model), ('gb', gb_model), ('svm', svm_model)],
    voting = 'soft')
voting_clf.fit(Xtrain, ytrain)

# Showing the individual classifiers and their predictions
for clf in (rnd_model, log_model, gb_model, svm_model, voting_clf):
    clf.fit(Xtrain, ytrain)
    y_pred_ind = clf.predict(Xtest)
    print(clf.__class__.__name__, accuracy_score(ytest, y_pred_ind))
print()

# Making predictions using the Voting Classifier
y_pred = voting_clf.predict(Xtest)

# Using the Bagging Classifier to improve overall performance and generalization of the model
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_depth=20), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(Xtrain, ytrain)
y_pred_bag = bag_clf.predict(Xtest)
class_report = classification_report(ytest, y_pred_bag)
print(class_report)
print()

# Accuracy of the Voting Classifier's performance
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy}")
print()

# Computing the Confusion Matrix for the Voting Classifier's performance
con_matrix = confusion_matrix(ytest, y_pred)
print(f"Confusion Matrix: {con_matrix}")
print()

# Compute the ROC curve and AUC for the Random Forest Classifier
# Predict probabilities
mdl = rnd_model.fit(Xtrain, ytrain)
prob = mdl.predict_proba(Xtest)[:, 1]
# ROC curve and AUC
fpr, tpr, _ = roc_curve(ytest, prob)
roc_auc = auc(fpr, tpr)
print(f"ROC_AUC: {roc_auc}")
# Other metrics
precision_vc = precision_score(ytest, y_pred, zero_division=1)
print(f"Precision: {precision_vc}")
recall = recall_score(ytest, y_pred)
print(f"Recall: {recall}")
f1_score_vc = f1_score(ytest, y_pred)
print(f"f1_Score: {f1_score_vc}")
print()

# Plot the ROC_AUC curve
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('Receiver Operating Characteristic')
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.show()


# Feature Selection - we can rerun the models using only the top contributing models, if we like.
# Get feature importance for the 2 most accurate models
# Get feature importance from the trained Random Forest model
importance = rnd_model.feature_importances_
print("Feature Importance from the trained Random Forest model")
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title('Feature Importance: Random Forest Model')
plt.ylabel('Score')
plt.xlabel('Features')
plt.show()
print()

# Get feature importance from the trained Gradient Boosting model
importance = gb_model.feature_importances_
print("Feature Importance from the trained Gradient Boosting model")
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title('Feature Importance: Gradiant Boosting Model')
plt.ylabel('Score')
plt.xlabel('Features')
plt.show()
print()
'''
Based on these feature importance, we can choose or remove some features (feature selection) as needed for each model.
'''

# Create pair plot to show pairwise relationships
sns.pairplot(data, hue="churn", diag_kind="hist")
plt.show()

# Interpretation
'''
In summary, the results indicate that the classifiers were good when determining accuracy and ROC AUC. However they had 
trouble accurately classifying the minority class (Churn = 1). This is because the data is imbalanced.
The low recall for the Churn class reflects this discrepancy. The high precision and recall for the majority class 
demonstrate that the models are more accurate at predicting that class (No Churn = 0). 
'''







