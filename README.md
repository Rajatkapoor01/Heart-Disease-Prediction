# Heart Disease Prediction Using Machine Learning

## Project Overview
This project aims to create a machine learning model to predict whether a patient is affected by heart disease based on various health metrics. The model is developed using Python and Flask, enabling seamless deployment for healthcare professionals.

## Control FLOW DIAGRAM:
   ![Flow1](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/Screenshot%202025-04-03%20225212.png)
   ![Flow2](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/Screenshot%202025-04-03%20225228.png)
   <br>
## Steps to Develop the ML Model

1. **Collect the Data**: Gather relevant patient health data to form the dataset used for training the model.
   
2. **Import Libraries and Load the Data**: Utilize essential libraries such as Pandas, NumPy, and Scikit-learn to load and manipulate the data.
   
3. **Preprocess the Data**: Clean the dataset by handling missing values, outliers, and ensuring data quality for accurate predictions.
   
4. **Feature Engineering**: Analyze the data to identify and create relevant features that improve the model's performance.
   
5. **Model Creation**: Develop and train various machine learning models to predict heart disease. Implement techniques such as cross-validation to evaluate model performance.

6. **Save the Best Model**: Use libraries like `joblib` or `pickle` to save the model that performs best during evaluation for future use.

7. **Deployment**: Deploy the model using a Flask application. This includes setting up a web server to allow doctors to interact with the model through a user-friendly interface.

8. **Develop a Flask App**: 
   - Build a Flask application that integrates the saved best model to enable predictions based on new patient data.
   - Launch the local server app to facilitate user interaction and predictions in real-time.

## Key Features
- **User-friendly Interface**: The web application allows healthcare professionals to input patient data and receive immediate predictions regarding heart disease.
- **Scalability**: The model can be easily deployed to cloud platforms for broader access.
- **Early Diagnostics**: This application aids in early detection of heart disease, improving patient outcomes through timely intervention.
## CONCEPT USE
Sure! Here are the definitions of these medical terms:

1. **Age**: 
   - The number of years a person has lived. In the context of heart disease, age is a significant risk factor; the risk increases as age increases.

2. **Chest Pain Type**:
   - **Typical Angina**: Chest pain related to heart disease, usually triggered by physical activity or stress.
   - **Atypical Angina**: Chest pain not typically associated with heart disease, often caused by other conditions.
   - **Non-Anginal Pain**: Pain not related to heart disease.
   - **Asymptomatic**: No chest pain, even though heart disease might be present.

3. **Serum Cholesterol**:
   - A fat-like substance found in the blood. High levels of cholesterol, especially LDL ("bad" cholesterol), can increase the risk of heart disease.

4. **Maximum Heart Rate Achieved**:
   - The highest number of heartbeats per minute during physical activity. It can be used to assess the heart's health and fitness level.

5. **ST Depression**:
   - A downward shift of the ST segment on an ECG (electrocardiogram) that can indicate reduced blood flow to the heart, often due to ischemia (lack of oxygen).

6. **Number of Major Vessels Colored by Fluoroscopy**:
   - A measure of how many coronary arteries have blockages. A fluoroscopy is a type of imaging used to observe blood flow and identify blockages in coronary arteries.

7. **Thalassemia Status**:
   - A blood disorder that affects hemoglobin (the protein that carries oxygen in the blood). The term "fixed defect" refers to permanent damage, and "reversible defect" refers to temporary issues that can improve with treatment.
## Normal RANGE:
Here's a detailed table for all the parameters related to heart disease prediction, including the normal range and what each value might indicate regarding the likelihood of heart disease:

| **Parameter**                                  | **Value**                              | **Normal Range / Risk Interpretation**                                             |
|------------------------------------------------|----------------------------------------|-----------------------------------------------------------------------------------|
| **Age**                                        | Younger (<45 years for men, <55 years for women) | **Normal**: Low risk for heart disease.                                             |
|                                                | Older (≥45 years for men, ≥55 years for women) | **Higher Risk**: Increased risk of heart disease with age.                         |
| **Chest Pain Type**                            | **0** - Asymptomatic                    | **Normal**: No chest pain; lower risk of heart disease.                            |
|                                                | **1** - Atypical Angina                 | **Moderate Risk**: Pain not typical of heart disease; needs further evaluation.    |
|                                                | **2** - Non-Anginal Pain                | **Low Risk**: Pain not related to heart disease, could be due to other conditions. |
|                                                | **3** - Typical Angina                  | **High Risk**: Classic chest pain associated with heart disease.                   |
| **Serum Cholesterol (mg/dl)**                  | < 200 mg/dl                            | **Normal**: Desirable cholesterol level.                                           |
|                                                | 200-239 mg/dl                          | **Borderline High**: Risk of heart disease may be elevated.                        |
|                                                | ≥ 240 mg/dl                            | **High**: High cholesterol is a major risk factor for heart disease.               |
| **Maximum Heart Rate Achieved**                | Predicted HR = 220 - Age               | **Normal**: A heart rate close to the predicted value suggests good heart health.   |
|                                                | Below predicted HR                     | **Higher Risk**: A lower-than-predicted maximum heart rate may indicate heart issues. |
| **ST Depression (mm)**                         | 0 mm (no depression)                   | **Normal**: No ST depression, which indicates normal heart function.               |
|                                                | 0.5-1 mm                               | **Borderline**: Mild depression, could indicate minor ischemia.                    |
|                                                | ≥ 1 mm                                  | **High Risk**: Significant depression, suggesting ischemia or poor blood flow.     |
| **Number of Major Vessels Colored by Fluoroscopy** | 0 vessels                              | **Normal**: No blockages, healthy arteries.                                        |
|                                                | 1-2 vessels                            | **Moderate Risk**: Partial blockages, may require lifestyle changes or monitoring. |
|                                                | 3 vessels                              | **High Risk**: Severe blockages, likely requiring medical intervention (e.g., surgery). |
| **Thalassemia Status**                         | **0** - Normal                          | **Normal**: No thalassemia-related issues, lower risk.                             |
|                                                | **1** - Fixed Defect                    | **Moderate Risk**: Permanent damage to the heart muscle, often due to past heart attack. |
|                                                | **2** - Reversible Defect               | **Moderate Risk**: Temporary damage or ischemia that could improve with treatment. |

### Explanation of the Ranges:
1. **Age**: Older individuals are at higher risk for heart disease due to the gradual buildup of plaque in the arteries over time.
   
2. **Chest Pain Type**: Typical angina (chest pain) is a strong indicator of heart disease, while atypical pain or no chest pain at all (asymptomatic) usually lowers the risk. 

3. **Serum Cholesterol**: A cholesterol level above 200 mg/dl, especially >240 mg/dl, indicates a higher risk for heart disease due to plaque buildup in arteries.

4. **Maximum Heart Rate Achieved**: A lower-than-predicted maximum heart rate during exercise could indicate an underlying heart condition, such as poor cardiovascular fitness or ischemia.

5. **ST Depression**: A significant depression (≥1 mm) in the ST segment during an exercise ECG indicates reduced blood flow to the heart (ischemia), which is a sign of heart disease.

6. **Number of Major Vessels Colored by Fluoroscopy**: The greater the number of blocked coronary arteries, the higher the risk for heart disease. A blockage in 3 vessels often requires surgical intervention, such as bypass surgery.

7. **Thalassemia Status**: A "fixed defect" indicates permanent damage, often from previous heart attacks or chronic ischemia, while a "reversible defect" suggests temporary issues that may be improved with treatment.

This detailed table helps to better understand the correlation between various health metrics and heart disease risk.

## Getting Started
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd heart-disease-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```


5. Open your web browser and navigate to `http://localhost:5000` to access the application.
6. Here is the Input Screeen:
   ![alt text](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/images/img-1.png)
7. Here is the Input Screen:
    ![alt text](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/images/img-2.png)
8. Here you have your output screen.
   ![alt text](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/images/img-3.png)
   
## Conclusion
This project demonstrates the implementation of a machine learning model for heart disease prediction, with a focus on usability for healthcare professionals. The combination of data preprocessing, model training, and deployment using Flask ensures a practical solution for early diagnostics.

## FINAL OUTPUT :
![result 1](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/Screenshot%202024-11-06%20072910.png)

![result 2](https://github.com/Rajatkapoor01/Heart-Disease-Prediction/blob/main/Screenshot%202024-11-06%20065240.png)
 

## LINKEDIN ARTICAL:
https://www.linkedin.com/pulse/heart-disease-prediction-using-machine-learning-rajat-kapoor-qachc/?trackingId=bRaiHoVyQ%2BayMyOKP4u5oA%3D%3D

## MEDIUM ARTICAL :
https://medium.com/@rajat01kapoor/heart-disease-prediction-using-machine-learning-a106dd953c51

## GITHUB LINK :
https://github.com/Rajatkapoor01/Heart-Disease-Prediction
