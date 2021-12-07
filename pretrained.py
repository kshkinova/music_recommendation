import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score

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
    print('\n---------- Running Logistic... ----------')
    log_reg = pickle.load(open(save_path + 'log_reg.sav','rb')) # load model

    importance = log_reg.coef_[0]
    get_feature_importance(importance, train_data.columns, 'logreg_importance.png',)
    
    print('Accuracy on valid: ', log_reg.score(x_valid,y_valid))
    print('Accuracy on train: ', log_reg.score(x_train,y_train))
    
    
    # SVM
    print('\n---------- Running SVM... ----------')
    model = pickle.load(open(save_path + 'svm.sav','rb')) # load model

    importance = model.coef_[0]
    get_feature_importance(importance, train_data.columns, 'svm_importance.png')
    
    print('Accuracy on valid: ', model.score(x_valid,y_valid))
    print('Accuracy on train: ', model.score(x_train,y_train))
    

    # Decision Tree
    print('\n---------- Running Decision Tree... ----------')
    tree = pickle.load(open(save_path + 'tree.sav','rb')) # load model

    importance = tree.feature_importances_
    get_feature_importance(importance, train_data.columns, 'tree_importance.png')
    
    print('Accuracy on valid: ', tree.score(x_valid,y_valid))
    print('Accuracy on train: ', tree.score(x_train,y_train))
    

    # XgBoost
    print('\n---------- Running XgBoost... ----------')
    model = pickle.load(open(save_path + 'xgb_model.sav', 'rb'))

    importance = model.feature_importances_
    get_feature_importance(importance, train_data.columns, 'xgb_importance.png')

    print('Accuracy on valid: ', accuracy_score(y_valid, model.predict(x_valid)>0.5))
    print('Accuracy on train: ', accuracy_score(y_train, model.predict(x_train)>0.5))
    
    
    # LightGBM
    print('\n---------- Running LightGBM... ----------')
    model = pickle.load(open(save_path + 'lgbm_model.sav', 'rb'))

    importance = model.feature_importances_
    get_feature_importance(importance, train_data.columns, 'lgbm_importance.png')

    print('Accuracy on valid: ', np.mean((model.predict(x_valid, num_iteration=model.best_iteration_)>0.5)==y_valid))
    print('Accuracy on train: ', np.mean((model.predict(x_train, num_iteration=model.best_iteration_)>0.5)==y_train))
    
    
def get_feature_importance(importance,labels,save_file,save_dir='./plots/'):
    plt.figure(figsize=(15,10))
    plt.xlabel('features',fontsize=30)
    plt.ylabel('importance',fontsize=30)
    plt.xticks(rotation='90') # rotate labels
    plt.bar(labels, importance)
    plt.savefig(save_dir + save_file)

if __name__ == '__main__':
    main(dataset_path='./training_data/', save_path='./models/', dev=True)