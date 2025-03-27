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

models = [('LDA', LinearDiscriminantAnalysis()), 
          ('KNN', KNeighborsClassifier(n_neighbors=10)),
          ('Naive Bayes', GaussianNB()),
          ('Decision Tree', DecisionTreeClassifier()),
          ('SVM', SVC(C=2.5, max_iter=5000)),
          ('Random Forest', RandomForestClassifier()),
          ('AdaBoost', AdaBoostClassifier())
          ]

metrics = [('Accuracy', accuracy_score), 
           ('F1-Score', f1_score), 
           ('Precision', precision_score)]
# ('Recall', recall_score), 

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
    print(f'{trans_name}\n----------------\n')
    scaler = trans
    (X_tr, X_te) = (X_train, X_test) if trans_name == 'NoTransform' else (scaler.fit_transform(X_train), scaler.fit_transform(X_test))
    # Train the model
    # model = LogisticRegression(solver='newton-cg')
    for metr_name, metric in metrics:
        print(f'Metric: {metr_name}\n---------------\n')
        for mod_name, model in models:
            classifier = model
            classifier.fit(X_tr, Y_train)
            # Evaluation
            Y_pred = classifier.predict(X_te)
            if metr_name == 'Accuracy':
                result = metric(Y_pred, Y_test)
            else:
                result = metric(Y_pred, Y_test, average='macro')
            print(f'{mod_name}: {result*100}')

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


