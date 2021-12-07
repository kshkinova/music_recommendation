import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import f_classif, SelectKBest

def main(dataset_path, save_path, dev):
    train_data = pd.read_csv(dataset_path + 'train_data.csv')
    train_target = pd.read_csv(dataset_path + 'train_target.csv')

    valid_data = pd.read_csv(dataset_path + 'valid_data.csv')
    valid_target = pd.read_csv(dataset_path + 'valid_target.csv')

    x_train = train_data.to_numpy()
    y_train = train_target.to_numpy()
    n1, dim1 = y_train.shape
    y_train = y_train.reshape((n1,))

    x_valid = valid_data.to_numpy()
    y_valid = valid_target.to_numpy()
    n2, dim2 = y_valid.shape
    y_valid = y_valid.reshape((n2,))
    
    print('Starting training')

    # Logistic Regression
    log_reg = LogisticRegression(solver='lbfgs', multi_class='auto')
    print('\n---------- Running Logistic... ----------')
    log_reg.fit(x_train, y_train)

    pickle.dump(log_reg, open(save_path + 'log_reg.sav','wb')) # save model

    importance = log_reg.coef_[0]
    get_feature_importance(importance, train_data.columns, 'logreg_importance.png',)
    evaluate_model(log_reg,x_train,y_train,x_valid,y_valid)
    
    # SVM
    model = svm.LinearSVC(max_iter=1000)
    print('\n---------- Running SVM... ----------')
    model.fit(x_train,y_train)

    pickle.dump(model, open(save_path + 'svm.sav','wb')) # save model

    importance = model.coef_[0]
    get_feature_importance(importance, train_data.columns, 'svm_importance.png')
    print('SVM score on train: ', model.score(x_train,y_train))
    print('SVM score on valid: ', model.score(x_valid,y_valid))

    # Decision Tree
    tree = DecisionTreeClassifier(criterion='gini', max_depth=10,
                                max_leaf_nodes=84, min_samples_split=5)
    print('\n---------- Running Decision Tree... ----------')
    tree.fit(x_train, y_train)

    pickle.dump(tree, open(save_path + 'tree.sav','wb')) # save model

    importance = tree.feature_importances_
    get_feature_importance(importance, train_data.columns, 'tree_importance.png')
    evaluate_model(tree,x_train,y_train,x_valid,y_valid)
    print('DecisionTree score on train: ', tree.score(x_train,y_train))
    print('DecisionTree score on valid: ', tree.score(x_valid,y_valid))

    # Random Forest
    forest = RandomForestClassifier(class_weight='balanced',criterion='gini', 
                       max_depth=25, min_samples_split=10, n_estimators=500, n_jobs=-1)
    print('\n---------- Running Random Forest... ----------')
    forest.fit(x_train, y_train)

    pickle.dump(forest, open(save_path + 'forest.sav','wb')) # save model

    importance = forest.feature_importances_
    get_feature_importance(importance, train_data.columns, 'forest_importance.png')
    evaluate_model(forest,x_train,y_train,x_valid,y_valid)

    # XgBoost

    model = xgb.XGBModel(objective ='binary:logistic', learning_rate = 0.1, colsample_bytree = 0.5, max_depth = 10, alpha = 10, n_estimators = 50)
    print('\n---------- Running XgBoost... ----------')
    model.fit(x_train, y_train, eval_set = [(x_valid,y_valid)], eval_metric = 'auc')
    pickle.dump(model,open(save_path + 'xgb_model.sav', 'wb'))

    print(accuracy_score(y_valid, model.predict(x_valid)>0.5))
    print(accuracy_score(y_train, model.predict(x_train)>0.5))
    importance = model.feature_importances_
    get_feature_importance(importance, train_data.columns, 'xgb_importance.png')

    # LightGBM

    model = lgb.LGBMModel(learning_rate = 0.3, max_depth = 15, num_leaves = 250, objective = 'binary', n_estimators=200)
    print('\n---------- Running LightGBM... ----------')
    model.fit(x_train,y_train, eval_set = [(x_valid,y_valid)], eval_metric = 'auc')
    pickle.dump(model,open(save_path + 'lgbm_model.sav', 'wb'))

    # model = pickle.load(open('./predictions/lgbm_model.sav', 'rb'))
    print(np.mean((model.predict(x_valid, num_iteration=model.best_iteration_)>0.5)==y_valid))
    print(np.mean((model.predict(x_train, num_iteration=model.best_iteration_)>0.5)==y_train))
    importance = model.feature_importances_
    get_feature_importance(importance, train_data.columns, 'lgbm_importance.png')

    # PCA ICA KBest
    pca = PCA(n_components=10)
    ica = FastICA(n_components=10)
    kbest = SelectKBest(f_classif, k=15)

    feature_selection(pca, x_train, y_train, x_valid, y_valid)
    feature_selection(ica, x_train, y_train, x_valid, y_valid)
    feature_selection(kbest, x_train, y_train, x_valid, y_valid)

    
def get_feature_importance(importance,labels,save_file,save_dir='./plots/'):
    plt.figure(figsize=(15,10))
    plt.xlabel('features',fontsize=30)
    plt.ylabel('importance',fontsize=30)
    plt.xticks(rotation='90') # rotate labels
    plt.bar(labels, importance)
    plt.savefig(save_dir + save_file)

def feature_selection(model, x_train, y_train, x_valid, y_valid):
    selection = model.fit(x_train, y_train)
    x_train_new = selection.transform(x_train)
    x_valid_new = selection.transform(x_valid)

    model = lgb.LGBMModel(learning_rate = 0.3, max_depth = 15, num_leaves = 250, objective = 'binary', n_estimators=200)

    model.fit(x_train_new,y_train, eval_set = [(x_valid_new,y_valid)], eval_metric = 'auc')
    print(np.mean((model.predict(x_valid_new, num_iteration=model.best_iteration_)>0.5)==y_valid))
    print(np.mean((model.predict(x_train_new, num_iteration=model.best_iteration_)>0.5)==y_train))

def evaluate_model(model,x_train,y_train,x_valid,y_valid):
    print('Accuracy on train: ', model.score(x_train,y_train))
    print('Accuracy on valid: ', model.score(x_valid,y_valid))
    print('roc auc on train:', roc_auc_score(y_train, model.predict_proba(x_train)[:,1]))
    print('roc auc on valid:', roc_auc_score(y_valid, model.predict_proba(x_valid)[:,1]))

if __name__ == '__main__':
    main(dataset_path='./training_data/', save_path='./models/', dev=True)