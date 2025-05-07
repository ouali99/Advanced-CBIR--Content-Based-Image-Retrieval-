Advanced Content-Based Image Retrieval (CBIR) System

🌟 Features
Image Descriptors

GLCM (Gray Level Co-occurrence Matrix) - Extracts texture features
BiT (Biological Taxonomy) - Biologically-inspired feature extraction
Haralick Features - Advanced statistical texture analysis
Combined Descriptors - Fusion of multiple descriptors for better performance

Distance Metrics

Euclidean Distance
Manhattan Distance
Chebyshev Distance
Canberra Distance

Machine Learning Classifiers

Linear Discriminant Analysis (LDA)
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Decision Trees
Random Forest
AdaBoost

Two Operation Modes

Basic Mode - Traditional similarity-based image retrieval
Advanced Mode - ML-enhanced classification and retrieval with performance metrics

📋 Prerequisites

Python 3.6+
OpenCV
NumPy
Streamlit
scikit-image
scikit-learn
scipy
mahotas
PIL (Pillow)

🔧 Installation

Clone the repository:
bashgit clone https://github.com/yourusername/advanced-cbir-system.git
cd advanced-cbir-system

Install the required dependencies:
bashpip install -r requirements.txt

Prepare your image dataset:

Create an images directory in the project root
Organize images into subdirectories by category:
images/
├── category1/
│   ├── image1.jpg
│   └── image2.jpg
├── category2/
│   ├── image3.jpg
│   └── ...



Extract and save feature signatures:
bashpython data_processing.py

Launch the application:
bashstreamlit run app_ad.py


🚀 Usage
Basic Mode

Select "Basic" in the mode selection radio button
Choose a descriptor (GLCM or BiT)
Select a distance metric
Set the number of similar images to display
Upload a query image
View the most similar images from your dataset

Advanced Mode

Select "Advanced" in the mode selection radio button
Choose a descriptor (including combined descriptors)
Select a distance metric
Choose a machine learning algorithm
Upload a query image
View classification results and performance metrics
See similar images from the predicted class

📁 Project Structure
├── app.py                       # Basic CBIR Streamlit interface
├── app_ad.py                    # Advanced CBIR with machine learning
├── data_processing.py           # Dataset processing with multiple descriptors
├── descriptor.py                # Image descriptors implementation
├── distances.py                 # Distance metrics implementation
├── BiT.py                       # Biological Taxonomy descriptor (not included)
├── fineTuningModels.py          # Model tuning and optimization
├── spotCheckmodelTrainig.py     # Model evaluation and comparison
├── images/                      # Image dataset folder
├── signatures_glcm.npy          # Pre-computed GLCM features
├── signatures_bitdesc.npy       # Pre-computed BiT features
├── signatures_haralick_feat.npy # Pre-computed Haralick features
├── signatures_bit_glcm_haralick.npy # Pre-computed combined features
└── requirements.txt             # Project dependencies
🛠️ Technical Details
Image Descriptors

GLCM: Extracts 6 texture features using the Gray Level Co-occurrence Matrix.
BiT: A biological taxonomy-inspired approach for feature extraction.
Haralick: 13 statistical features calculated from the GLCM.
Combined Descriptors: Fusion of multiple feature types for comprehensive image representation.

Machine Learning Pipeline

Feature Extraction: Convert images to meaningful numerical features
Data Preprocessing: Handle missing values, scaling, and normalization
Model Selection: Choose from multiple classification algorithms
Hyperparameter Tuning: Optimize model parameters with GridSearchCV
Model Evaluation: Calculate accuracy, precision, recall, and F1 score
Prediction: Classify new images and find similar ones in the predicted class

Performance Considerations

Pre-computed features are stored for efficient retrieval
Data scaling and normalization options are available for improved model performance
GridSearchCV provides optimized hyperparameters for each classifier

📊 Model Selection
The system includes various classifiers, each with different strengths:

LDA: Works well when classes are linearly separable
KNN: Simple but effective for many image classification tasks
SVM: Powerful for complex datasets with properly tuned kernels
Decision Tree: Provides interpretable results
Random Forest: Robust ensemble method with good generalization
AdaBoost: Boosts performance by focusing on difficult examples

🌱 Future Development

Implement deep learning-based feature extraction (CNN features)
Add image segmentation for region-based CBIR
Develop relevance feedback mechanisms
Support for video retrieval
Mobile application integration

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

GLCM implementation based on scikit-image
Machine learning models from scikit-learn
Web interface built with Streamlit
Haralick features implemented with mahotas

📞 Contact
Ouali OULD BRAHAM - ouali.ouldbraham2@gmail.com
Project Link: https://github.com/ouali99/Advanced-CBIR--Content-Based-Image-Retrieval-