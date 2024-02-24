# This is a Python implementation of the EL-HIP+ hierarchical feature selection method, which selects a set of positive features preserving hierarchical information for each instance.
# For the purpose of knowledge discovery, the selected features for each instance will be aggregated as one set of selected features for the entire dataset (Algorithm 1).
# For the purpose of predictive performance evaluation, an eager learning classifier (i.e. Random Forests) was trained by the EL-HIP+-selected features for each testing instance. 
# Note that, as an eager learning-based supervised learning paradigm, the classifier training process does not consider the feature values of the target testing instance.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score, average_precision_score


def HIPP(X, Anc, Des, Term_list, Term_index):
    status={}
    for index_1 in range(len(Term_list)):
        status[Term_list[index_1]]='selected'
    for index_11 in range(len(Term_list)):
        if X[index_11]==float('1'):
            ancestor_string = Anc.get(Term_list[index_11])
            ancestor_list=ancestor_string.split('&')
            ancestor_list.remove('\n')
            for index_111 in range(len(ancestor_list)):
                status[ancestor_list[index_111]]='removed'
        else:
            status[Term_list[index_11]]='removed'
            descendent_string= Des.get(Term_list[index_11])
            descendent_list=descendent_string.split('&')
            descendent_list.remove('\n')
            for index_112 in range(len(descendent_list)):
                status[descendent_list[index_112]]='removed'
    selected_term_list=[]
    for index_2 in range(len(Term_list)):
        if status[Term_list[index_2]]=='selected':
            selected_term_list.append(Term_list[index_2])
    selected_term_index=[]
    for index_3 in range(len(selected_term_list)):
        selected_term_index.append(int(Term_index.get(selected_term_list[index_3])))

    return selected_term_index


dir_HPO = open(".../GO_features_original.txt", "r")
HPO_Matrix = dir_HPO.readlines()

dir_GO = open(".../hpo_labels_30P_nonredundant.txt", "r")
GO_Matrix = dir_GO.readlines()

Feature_Matrix=[]
for index_1 in range(1, len(HPO_Matrix)):
    Feature_Value=HPO_Matrix[index_1].split(',')[1:-1]
    Feature_Matrix.append(Feature_Value)

x2 = np.array(Feature_Matrix)
x = x2.astype(np.float)

Class_Matrix=[]
for index_2 in range(1,len(GO_Matrix)):
    Class_Value=GO_Matrix[index_2].split(',')[1:-1]
    Class_Matrix.append(Class_Value)

y2 = np.array(Class_Matrix)
y = y2.astype(np.float)

HPO_terms_list=GO_Matrix[0].split(',')[1:-1]
GO_terms_list=HPO_Matrix[0].split(',')[1:-1]

dir_ancestors = open(".../Ancestors_GO.txt", "r")
allLines_Ancestors = dir_ancestors.readlines()
HPO_Ancestor={}
for index_1 in range(len(allLines_Ancestors)):
    HPO_Ancestor[allLines_Ancestors[index_1].split('%')[0]]=allLines_Ancestors[index_1].split('%')[1]

dir_descendent = open(".../Descendent_GO.txt", "r")
allLines_Descendents = dir_descendent.readlines()
HPO_Descendents={}
for index_1 in range(len(allLines_Descendents)):
    HPO_Descendents[allLines_Descendents[index_1].split('%')[0]]=allLines_Descendents[index_1].split('%')[1]

output_MCC = open(".../Results_HIPP_MCC_T30.txt", "a")
output_F1 = open(".../Results_HIPP_F1_T30.txt", "a")
output_AP = open(".../Results_HIPP_AP_T30.txt", "a")
output_selected_GOTerms = open(".../Selected_GOTerms.txt", "a")

MatrixHPOTerms=[]
for index_1 in range(len(HPO_Matrix)):
    Feature_Value_terms=HPO_Matrix[index_1].split(',')
    MatrixHPOTerms.append(Feature_Value_terms)
    print()
MatrixHPOTerms_2 = np.array(MatrixHPOTerms)

Term_list=[]
Term_index={}
for index_1 in range(1,len(MatrixHPOTerms_2[0])-1):
    Term_index[MatrixHPOTerms_2[0][index_1]] = index_1-1
    Term_list.append(MatrixHPOTerms_2[0][index_1])

for index_1 in range(len(y[0])):
    output_selected_GOTerms.write(HPO_terms_list[index_1]+'&')
    class_vector=y[:,index_1]
    MCC_CV=[]
    ap_CV=[]
    f1_CV=[]
    skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
    for train, test in skf.split(x, class_vector):
        Prediction_binary = []
        Prediction_prob = []
        for index_2 in range(len(x[test])):
            x_lazy = x[test][index_2]
            selected_term_index_HIP = HIPP(x_lazy, HPO_Ancestor, HPO_Descendents, Term_list, Term_index)
            X_sel_test = x_lazy[selected_term_index_HIP]
            X_sel_train = x[train][:, selected_term_index_HIP]
            clf = RandomForestClassifier(random_state=10)
            clf.fit(X_sel_train, class_vector[train])
            feature_importance=clf.feature_importances_
            for index_3 in range(len(selected_term_index_HIP)):
                output_selected_GOTerms.write(Term_list[selected_term_index_HIP[index_3]]+'+'+str(feature_importance[index_3])+'%')
            Y_pred_lazy = clf.predict(np.array([X_sel_test]))
            Y_pred_lazy_prob = clf.predict_proba(np.array([X_sel_test]))
            if len(Y_pred_lazy_prob[0]) != 1:
                Prediction_prob.append(Y_pred_lazy_prob[0,1])
            else:
                Prediction_prob.append(float(0))
            Prediction_binary.append(Y_pred_lazy)

        mccTesting = matthews_corrcoef(class_vector[test], Prediction_binary)
        f1Testing = f1_score(class_vector[test], Prediction_binary)
        averagePrecision = average_precision_score(class_vector[test], Prediction_prob)
        MCC_CV.append(mccTesting)
        f1_CV.append(f1Testing)
        ap_CV.append(averagePrecision)

    output_MCC.write(str(np.mean(MCC_CV)) + '\n')
    output_F1.write(str(np.mean(f1_CV))+'\n')
    output_AP.write(str(np.mean(ap_CV))+'\n')
    output_selected_GOTerms.write('\n')

output_MCC.flush()
output_MCC.close()
output_F1.flush()
output_F1.close()
output_AP.flush()
output_AP.close()
output_selected_GOTerms.flush()
output_selected_GOTerms.close()
