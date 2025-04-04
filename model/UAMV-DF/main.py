from gcForest import mv_gcForest
from evaluation import accuracy,f1_binary,f1_macro,f1_micro
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, \
    roc_auc_score, average_precision_score
from imblearn.metrics import geometric_mean_score
import csv
import time
import pandas as pd
from sklearn.metrics import accuracy_score

def get_predict_report(y_test, y_pred, y_prob):
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report['accuracy']
    recall_0 = report["0"]['recall']
    recall_1 = report["1"]['recall']
    precision_0 = report["0"]['precision']
    precision_1 = report["1"]['precision']

    f1 = f1_score(y_test, y_pred)
    f1_ma = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred)
    recall_ma = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred)
    precision_ma = precision_score(y_test, y_pred, average='macro')
    g_mean = geometric_mean_score(y_test, y_pred, average='binary')
    g_mean_ma = geometric_mean_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_prob[:, 1])
    aupr = average_precision_score(y_test, y_prob[:, 1])

    pre_report = np.array(
        [acc, recall_0, recall_1, precision_0, precision_1, f1, f1_ma, recall, recall_ma, precision, precision_ma,
         g_mean, g_mean_ma, auc, aupr])

    return pre_report


def output_report(fold, report, feature_name, ss):
    header = ["acc", " recall_0", "recall_1", "precision_0", "precision_1", "f1", "f1_macro", "recall",
              "recall_macro", "precision", "precision_macro", "g_mean", "g_mean_macro", "auc", "aupr", "best_layer_id","time"]
    if fold == 0:
        with open(f"./{result_path}{feature_name}_{ss}.csv", mode='w', newline='', encoding='utf8') as cf:
            wf = csv.writer(cf)
            wf.writerow(header)
            wf.writerow(report)
    else:
        with open(f"./{result_path}{feature_name}_{ss}.csv", mode='a', newline='', encoding='utf8') as cf:
            wf = csv.writer(cf)
            wf.writerow(report)

    cf.close()

    return


def get_feature(feature_name):
    all_feature=None
    feature_poss=[0]

    l=feature_name.split('+')
    for feature_n in l:
        feature = np.load(img_feature_path + f"{feature_n}.npy")
        if all_feature is None:
            all_feature = np.copy(feature)
        else:
            all_feature = np.hstack([all_feature, np.copy(feature)])
        feature_poss.append(feature.shape[1]+feature_poss[-1])

    return all_feature,feature_poss


def get_config(view_num): # 无关内容要处理掉
    config = {}
    config["random_state"] = 0
    config["max_layers"] = 100
    config["early_stop_rounds"] = 3
    config["train_evaluation"] = f1_macro  ##f1_binary,f1_macro,f1_micro,accuracy

    config["uncertainty_threshold"]=0.15
    config["clf_name"]='catboost' #  "catboost" / "svc" / "xgboost" / "adaboost" / "lgbm" / "guassianNB" / "knn" /"gradientBoost" / "mlp"

    config["estimator_configs"] = []
    num_jobs = 30
    for i in range(view_num):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None,
             "n_jobs": num_jobs})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None,
             "n_jobs": num_jobs})

    return config

n_fold = 50
feature_name = 'phy+hist+dwt+gldm'
img_feature_path = '../hybrid features/'
n_fold_path = '../50_folds/'
result_path = './output/'

if __name__=='__main__':

    print("feature_name=", feature_name)
    feature, feature_poss = get_feature(feature_name)
    print(feature_poss)

    view_num = len(feature_poss) - 1

    config = get_config(view_num)

    for fold in range(n_fold):
        print("fold-", str(fold), "+++++++++++++++++++++++++++++++++++++++++++++\n\n")

        x_train_id = np.load(n_fold_path + f"train_fold{fold}_id.npy")
        x_test_id = np.load(n_fold_path + f"test_fold{fold}_id.npy")
        y_train = np.load(n_fold_path + f"train_fold{fold}_y.npy")
        y_test = np.load(n_fold_path + f"test_fold{fold}_y.npy")

        x_train = feature[x_train_id]
        x_test = feature[x_test_id]
        print("x_train shape", x_train.shape)
        print("x_test shape", x_test.shape)

        t1 = time.time()

        gcf = mv_gcForest(config)
        y_train_pre, y_train_prob, best_layer_id = gcf.fit(x_train, y_train, feature_poss, fold)

        y_pred, y_prob, opinion = gcf.predict(x_test, fold)
        t2 = time.time()

        train_report = get_predict_report(y_train, y_train_pre, y_train_prob)
        train_report = np.append(train_report, best_layer_id)
        train_report = np.append(train_report, t2 - t1)

        test_report = get_predict_report(y_test, y_pred, y_prob)
        test_report = np.append(test_report, best_layer_id)
        test_report = np.append(test_report, t2 - t1)

        output_report(fold, train_report, feature_name, "train")
        output_report(fold, test_report, feature_name, "test")



















