Heart Attack Probability Prediction
===================================

Project Overview
----------------

This project aims to predict the probability of a heart attack using the Heart Attack Probability dataset obtained from Kaggle. The dataset includes various health indicators of patients, and analyses have been conducted to determine the risk of heart attacks using machine learning models.

Dataset
-------

The dataset contains various health parameters that affect the probability of heart attacks. Below are the features in the dataset along with their descriptions:

### Features

*   **age**: Age of the patient
    
*   **sex**: Gender (1 = Male, 0 = Female)
    
*   **cp**: Chest pain type (0-3)
    
*   **trtbps**: Resting blood pressure (mm Hg)
    
*   **chol**: Serum cholesterol level (mg/dl)
    
*   **fbs**: Fasting blood sugar (> 120 mg/dl) (1 = Yes, 0 = No)
    
*   **restecg**: Resting electrocardiographic results (0, 1, 2)
    
*   **thalachh**: Maximum heart rate achieved
    
*   **exng**: Exercise induced angina (1 = Yes, 0 = No)
    
*   **oldpeak**: ST depression induced by exercise relative to rest
    
*   **slp**: Slope of the ST segment during exercise
    
*   **caa**: Number of major vessels colored by fluoroscopy (0-3)
    
*   **thall**: Thallium stress test result (0 = Normal, 1 = Fixed defect, 2 = Reversable defect)
    
*   **output**: Target variable (0 = Less probability of heart attack, 1 = More probability of heart attack)
    

Technologies
------------

This project has been developed using the following technologies and libraries:

*   **Python**: Programming language
    
*   **Pandas**: Data manipulation and analysis
    
*   **NumPy**: Numerical computations
    
*   **Seaborn & Matplotlib**: Data visualization
    
*   **Scikit-learn**: Machine learning models
    

## Installation and Usage

1.  **Install Required Libraries:** You can install the required Python libraries with the following command:

    ```bash
    pip install numpy pandas seaborn matplotlib scikit-learn
    ```

2.  **Load the Dataset:** Download the dataset from Kaggle and place it in your project directory.

3.  **Run Jupyter Notebook:** You can use the Jupyter Notebook environment to examine and run the project. If you want to run it on Colab, follow these steps:

    *   Go to Google Colab.
        
    *   Upload the downloaded .ipynb file to Colab.

    *   Load the files using the command: 
        `from google.colab import files; uploaded = files.upload()`

    *   Run all the cells in the notebook to perform analyses and modeling.

        

Conducted Analyses
------------------

The following steps have been followed in the project:

1.  **Data Loading and Examination**
    
    *   The dataset is loaded into a Pandas DataFrame and basic information is examined.
        
    *   The structure of the dataset, column types, and summary statistics are analyzed.
        
2.  **Data Exploration and Visualization**
    
    *   Numerical and categorical variables in the dataset have been identified.
        
    *   The distribution of variables such as gender and chest pain type has been visualized.
        
    *   The distribution of the data has been visualized using pie charts.
        
3.  **Data Preprocessing**
    
    *   Missing values have been checked and cleaned.
        
    *   Outliers have been examined and necessary cleaning operations have been performed.
        
    *   Categorical variables have been digitized using One-Hot Encoding.
        

Modeling and Results
--------------------

Various machine learning models have been applied to the dataset, and the most suitable model has been selected to predict the probability of heart attacks.

*   **Model Used**: K-Nearest Neighbors (KNN)
    
*   **Comparison with SVM**: The performance of KNN was compared with Support Vector Machines (SVM).
    

### Results

*   **Best Model**: K-Nearest Neighbors (KNN)
    
*   **Accuracy Rate**: 85.85%
    

The accuracy rate of the model shows that reliable results have been obtained in predicting the probability of heart attacks. The performance of the model can be further improved based on the characteristics of the dataset and the methods used.

Conclusion
----------

This project encompasses the process of predicting the probability of heart attacks using health data. The results obtained through data preprocessing, exploratory data analysis, and machine learning models provide significant insights into determining the risk of heart attacks. The achieved accuracy rate demonstrates the applicability of such models in the healthcare sector.
