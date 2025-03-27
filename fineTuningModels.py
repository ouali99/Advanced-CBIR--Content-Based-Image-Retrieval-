import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from descriptor import glcm, bitdesc
import os

# Linear Algorithms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Nonlinear Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Data Transformation
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, Binarizer
transforms = [('NoTransform', None),
              ('Rescale', MinMaxScaler()), 
              ('Normalization', Normalizer()),
              ('Standardization', StandardScaler())
              ]
#('Binarization', Binarizer(threshold=0.0))

models = [ ('KNN', KNeighborsClassifier)]
K_neighbors = [x for x in range(1, 30)]
metrics = [('Accuracy', accuracy_score)]
# ('Recall', recall_score), 
Acc_noTrans = list()
Acc_rescale = list()
Acc_norm = list()
Acc_standard = list()
# Load Signatures / Feature vector
load_signatures = np.load('iris_glcm.npy')
# Split inputs / outputs
X = load_signatures[ : , : -1].astype('float')
Y = load_signatures[ : , -1].astype('int')
# Define test proportion
train_proportion = 0.15
seed = 10
# Split train / test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_proportion, random_state=seed)
# Add transforms
for trans_name, trans in transforms:
    #print(f'{trans_name}\n----------------\n')
    scaler = trans
    (X_tr, X_te) = (X_train, X_test) if trans_name == 'NoTransform' else (scaler.fit_transform(X_train), scaler.fit_transform(X_test))
    # Train the model
    # model = LogisticRegression(solver='newton-cg')
    for metr_name, metric in metrics:
        #print(f'Metric: {metr_name}\n---------------\n')
        for mod_name, model in models:
            for k in K_neighbors:
                #print(f'Neighbors: {k}\n-------------')
                classifier = model(n_neighbors=k)
                classifier.fit(X_tr, Y_train)
                # Evaluation
                Y_pred = classifier.predict(X_te)
                if metr_name == 'Accuracy':
                    result = metric(Y_pred, Y_test)
                else:
                    result = metric(Y_pred, Y_test, average='macro')
               # print(f'{mod_name}: {result*100}')
                if trans_name =='NoTransform':
                    Acc_noTrans.append((k, result))
                elif trans_name =='Rescale':
                    Acc_rescale.append((k, result))
                elif trans_name == 'Normalization':
                    Acc_norm.append((k, result))
                elif trans_name =='Standardization':
                    Acc_standard.append((k, result))
                else:
                    pass
Acc_noTrans.sort(key=lambda x: x[1])
Acc_noTrans.reverse()
print(f'No trans Acc: {Acc_noTrans}')
# # Make prediction
# import cv2
# print('Predicting on new data\n----------------------')
# path = 'iris1.png'
# width = 256
# height = 256
# image_rel_path = os.path.join('Samples', 'Iris2.png')
# print(image_rel_path)

# img = cv2.imread(image_rel_path)

# try:
#     img = cv2.resize(img, (width, height))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     feat =  glcm(img)
#     print(f'I am here : {img.shape}')

#     new_data = np.array([feat])
#     pred = model.predict(new_data)
#     proba = model.predict_proba(new_data)
#     print(f'Class: {pred[0]}, Proba: {proba[0]}')
# except Exception as e:
#     print(f'Error: {e}')


