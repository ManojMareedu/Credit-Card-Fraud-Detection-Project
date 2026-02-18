
import mlflow 
import matplotlib.pyplot as plt
from zenml import step
from sklearn.metrics import auc, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from data_ingestion import load_data
from typing import Tuple
from sklearn.base import BaseEstimator


df = load_data('card_transdata.csv')


def train_test_split_data(df):

    var = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip',
       'used_pin_number', 'online_order']

    X = df[var]
    y = df['fraud']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_data(df)

def evaluation_metrics(y_test, y_pred):
    print('Precision Score :',precision_score(y_test, y_pred))
    print('Recall Score:',recall_score(y_test, y_pred))
    print('F1 Score:',f1_score(y_test, y_pred))
    print('Accuracy score:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n',confusion_matrix(y_test, y_pred))

    return precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), accuracy_score(y_test, y_pred)


#Logistic Regression
@step(experiment_tracker="mlflow_fresh")
def CreditFraud_logreg_model(X_train, y_train, X_test, y_test)-> Tuple[BaseEstimator, float]:

    mlflow.sklearn.autolog()
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Co-efficients of logistic model:',logreg.coef_)
    print('Intercept of logistic model:',logreg.intercept_)

    ## (b) Accuracy
    print("Test Accuracy for logistic model is:", logreg.score(X_test, y_test))
    y_test_log = logreg.predict(X_test)

    #Logistic regression evaluation metrics

    Precision, Recall, F1, Accuracy = evaluation_metrics(y_test, y_test_log)

    y_pred_proba = logreg.predict_proba(X_test)[:,1]

    logreg_AUC = roc_auc_score(y_test, y_pred_proba)
    print("AUC for logistic regression:", logreg_AUC)

    logreg_score = 0.5 * logreg_AUC + 0.5 * F1

    mlflow.log_metric("LogReg precision", Precision)
    mlflow.log_metric("LogReg recall", Recall)
    mlflow.log_metric("LogReg f1", F1)
    mlflow.log_metric("LogReg accuracy", Accuracy)
    mlflow.log_metric("LogReg auc", logreg_AUC)
    mlflow.log_metric("LogReg selection score", logreg_score)

    # ROC Curve
    from sklearn.metrics import roc_curve

    # Get Measures
    FPR, TPR, threshold = roc_curve(y_test, y_pred_proba)

    # Generate Figure (create explicitly for clarity)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(FPR, TPR)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Logistic Regression ROC Curve")
    ax.grid(True)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')  # Add diagonal for reference
    ax.legend()

    # Log and close (inside mlflow.start_run())
    mlflow.log_figure(fig, "roc_curve_logreg.png")
    plt.close(fig)  # Prevents memory issues in pipelines

    return logreg, logreg_score


#Knn Classifier
@step(experiment_tracker="mlflow_fresh")

def CreditFraud_knn_model(X_train, y_train, X_test, y_test)-> Tuple[BaseEstimator, float]:
    
    mlflow.sklearn.autolog()
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    # define function
    knn = KNeighborsClassifier()

    # define a list of parameters
    param_knn = {'n_neighbors': range(3, 8, 2)}   


    #apply grid search
    grid_knn = GridSearchCV(knn, param_knn, cv = 5)
    grid_knn.fit(X_train, y_train)

    # the best hyperparameter chosen:
    print('The Best hyperparameter choosen: ',grid_knn.best_params_)

    print('The best validation score',grid_knn.best_score_)

    y_test_knn = grid_knn.best_estimator_.predict(X_test)

    #Knn evaluation metrics
    knn_Precision, knn_Recall, knn_F1, knn_Accuracy = evaluation_metrics(y_test, y_test_knn)

    y_test_knnproba = grid_knn.best_estimator_.predict_proba(X_test)[:,1]
    knn_AUC = roc_auc_score(y_test, y_test_knnproba)
    print("AUC for KNN model:", knn_AUC)

    knn_score = 0.5 * knn_AUC + 0.5 * knn_F1

    mlflow.log_metric("KNN precision", knn_Precision)
    mlflow.log_metric("KNN recall", knn_Recall)
    mlflow.log_metric("KNN f1", knn_F1)
    mlflow.log_metric("KNN accuracy", knn_Accuracy)
    mlflow.log_metric("KNN auc", knn_AUC)
    mlflow.log_metric("KNN selection score", knn_score)
    # Get Measures
    FPR_knn, TPR_knn, threshold = roc_curve(y_test, y_test_knnproba)

    # Generate Figure (create explicitly for clarity)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(FPR_knn, TPR_knn)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("KNN ROC Curve")
    ax.grid(True)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')  # Add diagonal for reference
    ax.legend()

    # Log and close (inside mlflow.start_run())
    mlflow.log_figure(fig, "roc_curve_knn.png")
    plt.close(fig)  # Prevents memory issues in pipelines


    return grid_knn.best_estimator_, knn_score


#Decision Tree Classifier
@step(experiment_tracker="mlflow_fresh")


def CreditFraud_decision_tree_model(X_train, y_train, X_test, y_test)-> Tuple[BaseEstimator, float]:
    
    mlflow.sklearn.autolog()
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    opt_tree = DecisionTreeClassifier(random_state = 42)

    dt_params = {'max_depth':  range(1,10)}

    grid_tree = GridSearchCV(opt_tree, dt_params)
    grid_tree.fit(X_train, y_train)

    print('The Best hyperparameter choosen for Decision Tree: ',grid_tree.best_params_)

    y_pred_dt = grid_tree.predict(X_test)

    print('Test Accuracy of Decision Tree: ',grid_tree.score(X_test, y_test))
    print('Train Accuracy of Decision Tree: ',grid_tree.score(X_train, y_train))

    #Decision Tree evaluation metrics
    dt_Precision, dt_Recall, dt_F1, dt_Accuracy = evaluation_metrics(y_test, y_pred_dt)

    from sklearn import tree
    print(tree.export_text(grid_tree.best_estimator_))

    y_pred_proba_dt = grid_tree.predict_proba(X_test)[:,1]

    dt_AUC = roc_auc_score(y_test, y_pred_proba_dt)
    print("AUC for Decision Tree:", dt_AUC)

    dt_score = 0.5 * dt_AUC + 0.5 * dt_F1

    mlflow.log_metric(" DT precision", dt_Precision)
    mlflow.log_metric(" DT recall", dt_Recall)
    mlflow.log_metric(" DT f1", dt_F1)
    mlflow.log_metric(" DT accuracy", dt_Accuracy)
    mlflow.log_metric(" DT auc", dt_AUC)
    mlflow.log_metric("DT selection score", dt_score)
    # ROC Curve for Decision Tree

    FPR_dt, TPR_dt, threshold = roc_curve(y_test, y_pred_proba_dt)


    # Generate Figure (create explicitly for clarity)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(FPR_dt, TPR_dt)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Decision Tree ROC Curve")
    ax.grid(True)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')  # Add diagonal for reference
    ax.legend()

    # Log and close (inside mlflow.start_run())
    mlflow.log_figure(fig, "DT_ROC.png")
    plt.close(fig)  # Prevents memory issues in pipelines

    return grid_tree.best_estimator_, dt_score

#Random Forest Classifier
@step(experiment_tracker="mlflow_fresh")

def CreditFraud_random_forest_model(X_train, y_train, X_test, y_test)-> Tuple[BaseEstimator, float]:
    
    mlflow.sklearn.autolog()
    from sklearn.ensemble import RandomForestClassifier
    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(n_estimators=25, random_state=42)

    # Train the Random Forest classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    #Random Forest evaluation metrics
    rf_Precision, rf_Recall, rf_F1, rf_Accuracy = evaluation_metrics(y_test, y_pred)

    y_pred_train = rf_classifier.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Accuracy train:", accuracy_train)

    y_pred_proba = rf_classifier.predict_proba(X_test)[:,1]


    # we have obtained the predicted probability in the previous step: y_pred_proba

    rf_AUC = roc_auc_score(y_test, y_pred_proba)
    print("AUC for random forests:", rf_AUC)

    rf_score = 0.5 * rf_AUC + 0.5 * rf_F1

    mlflow.log_metric("RF precision", rf_Precision)
    mlflow.log_metric("RF recall", rf_Recall)
    mlflow.log_metric("RF f1", rf_F1)
    mlflow.log_metric("RF accuracy", rf_Accuracy)
    mlflow.log_metric("RF auc", rf_AUC)
    mlflow.log_metric("RF selection score", rf_score)

    # Predicted Probability: y_pred_proba
    # Get Measures
    FPR_rf, TPR_rf, threshold = roc_curve(y_test, y_pred_proba)

 
    # Generate Figure (create explicitly for clarity)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(FPR_rf, TPR_rf)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Random Forest ROC Curve")
    ax.grid(True)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')  # Add diagonal for reference
    ax.legend()

    # Log and close (inside mlflow.start_run())
    mlflow.log_figure(fig, "roc_curve_rf.png")
    plt.close(fig)  # Prevents memory issues in pipelines

    print(rf_classifier.feature_importances_)
    return rf_classifier, rf_score

@step(experiment_tracker="mlflow_fresh")
def select_best_model(
    logreg_model, logreg_score,
    knn_model, knn_score,
    dt_model, dt_score,
    rf_model, rf_score
):
    models = {
        "logreg": (logreg_model, logreg_score),
        "knn": (knn_model, knn_score),
        "decision_tree": (dt_model, dt_score),
        "random_forest": (rf_model, rf_score),
    }

    best_name, (best_model, best_score) = max(
        models.items(),
        key=lambda x: x[1][1]
    )

    mlflow.log_param("best_model_name", best_name)
    mlflow.log_metric("best_model_selection_score", best_score)

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="fraud_detector"
    )

    return best_name



