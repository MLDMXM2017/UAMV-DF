from logger import get_logger
from layer import layer
from k_fold_wrapper import KFoldWapper
from uncertainty import joint_multi_opinion, opinion_to_proba
import csv
from sklearn.svm import SVC,LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import math
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

LOGGER=get_logger("gcForest")

class mv_gcForest(object):
    def __init__(self,config):
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.train_evaluation = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.layers = []
        self.uncertainty_threshold=config["uncertainty_threshold"]
        self.clf_list=[None]
        self.scaler_list=[None]
        self.clf_name=config['clf_name']

        self.view_opinion_generation_method = config.get("view_opinion_generation_method", 'mean')


    def fit(self, x_train, y_train, feature_poss, fold):
        x_train, n_feature, n_label = self.preprocess(x_train, y_train, feature_poss)
        self.estimator_per_view=len(self.estimator_configs)/self.n_view

        evaluate = self.train_evaluation

        best_layer_id = 0
        depth = 0
        best_layer_evaluation = 0.0
        best_layer_label_temp = None
        best_layer_prob = None

        # augmented features
        enhanced_feature = None

        while depth < self.max_layers:
            layer_all_opinions = {v: [] for v in range(self.n_view)}

            y_train_probas = np.zeros((x_train.shape[0], n_label * len(self.estimator_configs)))
            y_train_opinions = np.zeros((x_train.shape[0], 3 * len(self.estimator_configs)))

            current_layer = layer(depth)
            LOGGER.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(
                    current_layer.layer_id))
            LOGGER.info("The shape of x_train is {}".format(x_train.shape))

            another_y_train_probas_avg = np.zeros((x_train.shape[0], n_label))

            for index in range(len(self.estimator_configs)):
                config = self.estimator_configs[index].copy()
                k_fold_est = KFoldWapper(current_layer.layer_id, index, config, random_state=self.random_state)

                f_pos = int(index / self.estimator_per_view)
                x_train_feature = x_train[:, self.feature_poss[f_pos]:self.feature_poss[f_pos + 1]]
                if depth != 0:
                    x_train_feature = np.hstack((x_train_feature, enhanced_feature))

                # calculate predicted probability vector of RF and ET
                y_prob = k_fold_est.fit(x_train_feature, y_train)

                current_layer.add_est(k_fold_est)

                evidence = self.get_evidence_base_proba(y_prob)
                opinion = self.get_opinion_base_evidence(evidence)

                layer_all_opinions[f_pos].append(opinion)


                y_train_probas[:, index * n_label:index * n_label + n_label] += y_prob
                y_train_opinions[:, index * 3:index * 3 + 3] += opinion

                another_y_train_probas_avg += y_prob

            # calculate opinion for each view
            view_opinion_dict = {v: [] for v in range(self.n_view)}
            for v in range(self.n_view):  # 处理opinion
                cur_view_opinion_list = layer_all_opinions[v]
                view_opinion_dict[v].append(self.generate_view_opinion(
                    cur_view_opinion_list,
                    self.view_opinion_generation_method,
                ))

            # joint the opinion of multiple views
            joint_opinion_cur_layer_origin = joint_multi_opinion(
                [view_opinions[-1] for view_opinions in view_opinion_dict.values()])

            # enhanced hard sample learning module
            if depth>=1:

                hard_idx,hard_clf_opinion=self.reclassify_hard_sample(joint_opinion_cur_layer_origin,x_train,y_train)
                hard_df_opinion=joint_opinion_cur_layer_origin[hard_idx]

                hard_final_opinion=joint_multi_opinion([hard_clf_opinion,hard_df_opinion])

                joint_opinion_cur_layer_origin[hard_idx] = hard_final_opinion

            # calculate layer_prob
            y_train_probas_avg = opinion_to_proba(joint_opinion_cur_layer_origin)
            label_tmp = self.category[np.argmax(y_train_probas_avg, axis=1)]
            current_evaluation = evaluate(y_train, label_tmp)

            # renew the best layer
            if current_evaluation > best_layer_evaluation:
                best_layer_id = current_layer.layer_id
                best_layer_evaluation = current_evaluation
                best_layer_label_temp = label_tmp
                best_layer_prob = y_train_probas_avg
            LOGGER.info(
                "The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__, depth, current_evaluation))

            self.layers.append(current_layer)

            # gain no improvement within early_stop_rounds
            if current_layer.layer_id - best_layer_id >= self.early_stop_rounds:
                self.layers = self.layers[0:best_layer_id + 1]
                LOGGER.info("training finish...")
                LOGGER.info(
                    "best_layer: {}, current_layer:{}, save layers: {}".format(best_layer_id, current_layer.layer_id,
                                                                               len(self.layers)))
                break

            # Concatenate augmented features
            enhanced_feature = np.copy(y_train_probas)
            enhanced_feature = np.hstack([enhanced_feature, y_train_opinions])

            if depth>=1:
                single_prob = opinion_to_proba(hard_clf_opinion)
                hard_probs,hard_opinion=None,None
                for index in range(len(self.estimator_configs)):
                    if hard_probs is None:
                        hard_probs=np.copy(single_prob)
                        hard_opinion=np.copy(hard_clf_opinion)
                    else:
                        hard_probs=np.hstack([hard_probs, single_prob])
                        hard_opinion=np.hstack([hard_opinion, hard_clf_opinion])

                hard_enhanced_feature=np.hstack([hard_probs,hard_opinion])
                enhanced_feature[hard_idx]=hard_enhanced_feature


            self.store_layer_opinion(depth, view_opinion_dict, joint_opinion_cur_layer_origin, "train", fold)

            depth += 1

        return best_layer_label_temp, best_layer_prob, best_layer_id

    def predict(self, x,fold):
        prob, opinion = self.predict_opinion(x,fold)
        label = self.category[np.argmax(prob, axis=1)]
        return label, prob, opinion

    def predict_opinion(self, x_test,fold):
        n_label = len(self.category)
        enhanced_feature=None

        for depth in range(len(self.layers)):
            LOGGER.info(
                "--------------------------------------test_layer-{}--------------------------------------------".format(
                    depth))

            LOGGER.info("The shape of x_test is {}".format(x_test.shape))
            layer_all_opinions = {v: [] for v in range(self.n_view)}

            y_test_probas = np.zeros((x_test.shape[0], n_label * len(self.estimator_configs)))
            y_test_opinions = np.zeros((x_test.shape[0], 3 * len(self.estimator_configs)))

            for est_index in range(len(self.layers[depth].estimators)):
                f_pos = int(est_index / self.estimator_per_view)
                x_test_feature = x_test[:, self.feature_poss[f_pos]:self.feature_poss[f_pos + 1]]
                if depth != 0:
                    x_test_feature = np.hstack((x_test_feature, enhanced_feature))

                y_prob = self.layers[depth].estimators[est_index].predict_proba(x_test_feature)

                evidence = self.get_evidence_base_proba(y_prob)
                opinion = self.get_opinion_base_evidence(evidence)

                # RF or ET opinions within view
                layer_all_opinions[f_pos].append(opinion)

                # 保存一下当前行的prob
                y_test_probas[:, est_index * n_label:est_index * n_label + n_label] += y_prob
                y_test_opinions[:, est_index * 3:est_index * 3 + 3] += opinion

            view_opinion_dict = {v: [] for v in range(self.n_view)}
            for v in range(self.n_view):  # 处理opinion
                cur_view_opinion_list = layer_all_opinions[v]
                view_opinion_dict[v].append(self.generate_view_opinion(
                    cur_view_opinion_list,
                    self.view_opinion_generation_method,
                ))

            joint_opinion_cur_layer_origin = joint_multi_opinion(
                [view_opinions[-1] for view_opinions in view_opinion_dict.values()])

            if depth>=1:

                hard_idx, hard_clf_opinion = self.reclassify_test_hard_sample(depth,joint_opinion_cur_layer_origin, x_test)
                hard_df_opinion = joint_opinion_cur_layer_origin[hard_idx]

                hard_final_opinion = joint_multi_opinion([hard_clf_opinion, hard_df_opinion])

                joint_opinion_cur_layer_origin[hard_idx] = hard_final_opinion


            y_test_probas_avg = opinion_to_proba(joint_opinion_cur_layer_origin)



            enhanced_feature = np.copy(y_test_probas)
            enhanced_feature = np.hstack([enhanced_feature, y_test_opinions])
            if depth>=1:
                single_prob = opinion_to_proba(hard_clf_opinion)
                hard_probs,hard_opinion=None,None
                for index in range(len(self.estimator_configs)):
                    if hard_probs is None:
                        hard_probs=np.copy(single_prob)
                        hard_opinion=np.copy(hard_clf_opinion)
                    else:
                        hard_probs=np.hstack([hard_probs, single_prob])
                        hard_opinion=np.hstack([hard_opinion, hard_clf_opinion])

                hard_enhanced_feature=np.hstack([hard_probs,hard_opinion])
                enhanced_feature[hard_idx]=hard_enhanced_feature

        return y_test_probas_avg, joint_opinion_cur_layer_origin


    def preprocess(self,x_train,y_train,feature_poss):
        x_train=x_train.reshape((x_train.shape[0],-1))
        category=np.unique(y_train)
        self.category=category
        self.feature_poss=feature_poss
        self.n_class=len(category)
        self.n_view = len(feature_poss) - 1

        n_feature=x_train.shape[1]
        n_label=len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(x_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        return x_train,n_feature,n_label

    def get_evidence_base_proba(self,y_prob):
        n_class = self.n_class
        n_sample=y_prob.shape[0]
        evidence = np.empty(shape=(n_sample, n_class))

        e_0=[math.exp(x*2*self.n_class) for x in y_prob[:,0]]
        e_1=[math.exp(x*2*self.n_class) for x in y_prob[:,1]]

        evidence[:, 0] = e_0
        evidence[:, 1] = e_1

        return evidence


    def get_opinion_base_evidence(self,evidence,W=0.02):

        e_list=np.zeros((len(evidence), 1)) + 2
        opinion = np.hstack([evidence, e_list])

        opinion = opinion / np.sum(opinion, axis=1, keepdims=True)
        return opinion

    def generate_view_opinion(self, opinion_list, view_opinion_generation_method):

        if view_opinion_generation_method == "mean":  # 平均node产出的opinion
            opinion_view = np.mean(opinion_list, axis=0)
        return opinion_view

    def reclassify_hard_sample(self,train_opinion,x_train,y_train):
        uncertainty_l=train_opinion[:,2:3]
        hard_sample_idx=[idx for idx in range(x_train.shape[0]) if uncertainty_l[idx]>=self.uncertainty_threshold]
        easy_sample_idx=[idx for idx in range(x_train.shape[0]) if uncertainty_l[idx]<=self.uncertainty_threshold]
        clf=self.get_clf(self.clf_name)

        scaler = StandardScaler()
        easy_x_train = scaler.fit_transform(x_train[easy_sample_idx])
        hard_x_test = scaler.transform(x_train[hard_sample_idx])

        clf.fit(easy_x_train, y_train[easy_sample_idx])
        y_prob=clf.predict_proba(hard_x_test)

        hard_evidence = self.get_evidence_base_proba(y_prob)
        hard_opinion = self.get_opinion_base_evidence(hard_evidence)

        self.clf_list.append(clf)
        self.scaler_list.append(scaler)

        return hard_sample_idx, hard_opinion

    def reclassify_test_hard_sample(self, depth,test_opinion, x_test):
        uncertainty_l = test_opinion[:, 2:3]
        hard_sample_idx = [idx for idx in range(x_test.shape[0]) if uncertainty_l[idx] >= self.uncertainty_threshold]

        hard_x_test = self.scaler_list[depth].transform(x_test[hard_sample_idx])
        y_prob =  self.clf_list[depth].predict_proba(hard_x_test)

        hard_evidence = self.get_evidence_base_proba(y_prob)
        hard_opinion = self.get_opinion_base_evidence(hard_evidence)

        return hard_sample_idx, hard_opinion

    def get_clf(self,clf_name):
        clf=None
        n_jobs=-1

        if clf_name=='svc':
            clf = SVC(random_state=0, kernel='linear',probability=True)

        if clf_name=='catboost':
            clf = CatBoostClassifier(random_state=0)

        if clf_name=='xgboost':
            clf=XGBClassifier(random_state=0)

        if clf_name=='adaboost':
            clf=AdaBoostClassifier(random_state=0)

        if clf_name=='lgbm':
            clf=LGBMClassifier(random_state=0, n_jobs=n_jobs)

        if clf_name=='guassianNB':
            clf=GaussianNB()

        if clf_name=='knn':
            clf=KNeighborsClassifier(n_jobs=n_jobs)

        if clf_name=='gradientBoost':
            clf=GradientBoostingClassifier(random_state=0)

        if clf_name=='mlp':
            clf=MLPClassifier(random_state=0)

        return clf


    def store_layer_opinion(self,depth, view_opinion_dict, joint_opinion_cur_layer_origin, train_or_test, fold):
        # 记录四个视图上的opinion
        for view_idx in view_opinion_dict.keys():
            opinion_arr=view_opinion_dict[view_idx]

            opinion_df=pd.DataFrame(opinion_arr[0])
            opinion_df.columns=['belief_0','belief_1','uncertainty']
            opinion_df.to_csv(f'{train_or_test}_opinion/fold{fold}_layer{depth}_view{view_idx}.csv', index=False)

        # 记录joint的opinion
        opinion_df=pd.DataFrame(joint_opinion_cur_layer_origin)
        opinion_df.columns = ['belief_0', 'belief_1', 'uncertainty']
        opinion_df.to_csv(f'{train_or_test}_opinion/fold{fold}_layer{depth}_joint_layer.csv', index=False)

        return







