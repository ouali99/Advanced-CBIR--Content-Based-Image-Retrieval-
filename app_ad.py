# Import necessary libraries
import numpy as np
import os
import cv2
from descriptor import glcm, bitdesc, haralick_feat, bit_glcm_haralick
from distances import manhattan, euclidean, chebyshev, canberra
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import streamlit as st
from sklearn.model_selection import GridSearchCV


# List of descriptors functions
descriptors = {
    'GLCM': glcm,
    'BiT': bitdesc,
    'Haralick': haralick_feat,
    'BiT+Haralick+Glcm': bit_glcm_haralick
}

# List of distances
distance_functions = {
    'Euclidean': euclidean,
    'Manhattan': manhattan,
    'Chebyshev': chebyshev,
    'Canberra': canberra
}

# Load precomputed features
signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
signatures_haralick_feat = np.load('signatures_haralick_feat.npy', allow_pickle=True)
signatures_bit_glcm_haralick = np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)
features_db = {
    'GLCM': signatures_glcm,
    'BiT': signatures_bitdesc,
    'Haralick': signatures_haralick_feat,
    'BiT+Haralick+Glcm': signatures_bit_glcm_haralick
}

def find_similar_images(features, feature_db, distance_func, num_results):
    distances = [distance_func(features, db_feature[:-3]) for db_feature in feature_db]
    sorted_indices = np.argsort(distances)[:num_results]
    return sorted_indices

def extract_features(image, descriptor):
    descriptor_function = descriptors[descriptor]
    return descriptor_function(image)

def resize_image(img, width, height):
    return cv2.resize(img, (width, height))

def cbir():
    # Streamlit interface
    st.title('Content-Based Image Retrieval (CBIR)')

    # Sidebar for user input
    st.sidebar.header('Bar de recherche :')
    descriptor_choice = st.sidebar.selectbox("Choisissez un descripteur", ['GLCM', 'BiT'])
    distance_choice = st.sidebar.selectbox("Choisissez une mesure de distance", ['Euclidean', 'Manhattan', 'Chebyshev', 'Canberra'])
    num_results = st.sidebar.slider("Nombre d'images similaires à afficher", 1, 20, 5)

    # Centered uploader
    st.markdown("<div class='center-upload'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = image.convert('L')
        img_array = np.array(image.resize((256, 256)))  
   
        descriptor_func = descriptors[descriptor_choice]
        uploaded_image_features = descriptor_func(img_array)
   
        if descriptor_choice == 'GLCM':
            signatures = signatures_glcm
        else:
            signatures = signatures_bitdesc
   
    # Calculate distances between the uploaded image and dataset images
        distances = []
        dist_func = distance_functions[distance_choice]
   
        for signature in signatures:
            feature_vector = signature[:-3]  # Last two items are folder_name and relative_path
            dist = dist_func(uploaded_image_features, feature_vector)
            distances.append((dist, signature[-3], signature[-3]))  # distance, folder_name, relative_path
   
    # Sort distances
        distances.sort(key=lambda x: x[0])
   
        # Display top N similar images
        st.header(f"Top {num_results} images similaires")
        cols = st.columns(4)
        for i in range(num_results):
            dist, folder_name, relative_path = distances[i]
            img_path = os.path.join('images', relative_path)
            similar_img = Image.open(img_path)
            cols[i % 4].image(similar_img, caption=f"{folder_name}", use_column_width=True)
    else:
        st.write("Veuillez téléverser une image pour commencer.")


def cbirAdvanced():
    st.title('Content-Based Image Retrieval (CBIR) - Mode avancé')

    st.sidebar.header('Bar de recherche :')
    descriptor_choice = st.sidebar.selectbox("Choisissez un descripteur", list(descriptors.keys()))
    distance_choice = st.sidebar.selectbox("Choisissez une mesure de distance", list(distance_functions.keys()))
    num_results = st.sidebar.slider("Nombre d'images similaires à afficher", 1, 20, 5)
    
    st.write('Choisissez un algorithme de classification :')
    algorithm_choice = st.selectbox('', ['LDA', 'KNN', 'Decision Tree', 'SVM', 'Random Forest', 'AdaBoost'])

    # Centered uploader
    st.markdown("<div class='center-upload'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image = np.array(image)
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        uploaded_file_features = extract_features(image, descriptor_choice)

        signatures = features_db[descriptor_choice]

        X = np.array([sig[:-3] for sig in signatures], dtype=float)
        Y = np.array([sig[-1] for sig in signatures], dtype=int)
        
        # Replace inf values with finite values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)

        train_proportion = 0.15
        seed = 10
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=train_proportion, random_state=seed)
        # Define models
        models = {
            'LDA': LinearDiscriminantAnalysis(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'Random Forest': RandomForestClassifier(),
            'AdaBoost': AdaBoostClassifier()
        }

        # Define parameter grids for GridSearchCV
        param_grids = {
            'LDA': {
                'solver': ['svd', 'lsqr', 'eigen']
            },
            'KNN': {
                'n_neighbors': [1,20],
                'weights': ['uniform', 'distance']
            },
            'Decision Tree': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'criterion': ['gini', 'entropy']
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.001, 0.01, 0.1, 1]
            }
        }

        # Get the model
        model = models[algorithm_choice]

        # Perform Grid Search
        grid_search = GridSearchCV(model, param_grids[algorithm_choice], cv=5)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        result = best_model.score(x_test, y_test)
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        st.write('Best parameters:', grid_search.best_params_)
        st.write('Accuracy:', result)
        st.write('Accuracy2:', accuracy)
        st.write('Recall:', recall)
        st.write('F1:', f1)
        st.write('Precision:', precision)

        uploaded_file_features = np.array(uploaded_file_features).reshape(1, -1)
        uploaded_file_prediction = best_model.predict(uploaded_file_features)
        st.write('Prediction for uploaded image:', uploaded_file_prediction[0])

        # Find and display similar images
        similar_images_indices = [i for i, y in enumerate(Y) if y == uploaded_file_prediction[0]]
        similar_images = similar_images_indices[:num_results]
       
        st.write(f'Similar images to the uploaded image in class {uploaded_file_prediction[0]}:')
        cols = st.columns(4)
        for i, idx in enumerate(similar_images):
            image_path = os.path.join('./images', signatures[idx][-3])
            img = Image.open(image_path)
            cols[i % 4].image(img, use_column_width=True)


# Create a radio box to switch between cbir and cbirAdvanced
cbir_option = st.radio("Choose CBIR Mode:", ("Basic", "Advanced"))

if cbir_option == "Basic":
    cbir()
else:
    cbirAdvanced()

