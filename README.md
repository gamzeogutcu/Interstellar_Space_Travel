## 🚀 Interstellar Space Travel Project 🚀
# 1. Introduction  
  
Interstellar travel has long captured the imagination of humanity. With technological advancements, this dream is becoming a reality. The Interstellar Space Travel project aims to understand and predict customer satisfaction in the emerging field of interstellar tourism and travel.  
# 2. Project Objective  
  
The primary objective of this project is to understand and predict the Customer Satisfaction Score, a key indicator of service quality and customer experience in interstellar tourism. This score is crucial for future service improvements and customer loyalty strategies.  
# 3. Technologies Used  
  
   - Pycharm: For development and coding.  
   - Colab: For collaborative coding and data analysis.  
   - Streamlit: For creating interactive web applications.  
  
# 4. Methodology  
# Data Collection  
  
The dataset was sourced from Kaggle, containing information from surveys and feedback from customers who have experienced interstellar travel. The complete dataset consists of approximately 500,000 samples. For this analysis, a subset of 50,000 samples was used. This data includes demographic information, travel details, and customer feedback.
# Data Preprocessing & Exploratory Data Analysis (EDA)  
Data preprocessing involved several steps to ensure the data was clean and suitable for analysis:
   - Data Cleaning: Correcting any inconsistencies in the data.
   - Type Conversions: Ensuring that all data types were appropriately converted (e.g., converting categorical variables to datetime formats).
  
Exploratory Data Analysis (EDA) was performed to understand the underlying patterns and relationships in the data. The key steps involved were:  
   - Descriptive Statistics: Summarizing the main characteristics of the dataset.  
   - Distribution Analysis: Examining the distribution of customer satisfaction scores.  
   - Correlation Analysis: Identifying relationships between different variables.  
   - Visualization: Creating various plots to illustrate the data, such as histograms, box plots, and scatter plots.

# Feature Engineering
The steps of feature engineering involved were:  
   - Outlier Analysis: Identifying and handling anomalies in the data using IQR (Interquartile Range) technique. This step helped in improving the model's accuracy by focusing on the relevant data.  
   - Missing value analysis: identifying and determining how to handle empty or missing values within a dataset to ensure accurate modeling and analysis.  
   - Creating new features: deriving new information or features from existing variables within a dataset.  
   - Encoding: process of converting categorical or text data into a numerical format suitable for machine learning algorithms or other analytical methods.  
   - Normalization: Normalizing the data to bring all variables to a similar scale.  

# Model Development  
    
Several machine learning models were developed to predict customer satisfaction scores. These models included:    
    
  - Linear Regression  
  - Decision Trees  
  - Random Forest  
  - Ridge  
  - Lasso  
  - ElasticNet  
  - KNN  
  - Classification and Regression Trees (CART)  
  - Gradient Boosting Machines (GBM)  
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - Support Vector Machines (SVM)  
  
# Model Evaluation
  
The models were evaluated using metrics such as Root Mean Squared Error (RMSE), Root Mean Squared Error (RMSE), and R-squared. 

RMSE values are as follows:  
- RMSE: 2.8047 (LR) 
- RMSE: 2.8048 (Ridge) 
- RMSE: 5.288 (Lasso) 
- RMSE: 5.1343 (ElasticNet) 
- RMSE: 4.1973 (KNN) 
- RMSE: 4.3441 (CART) 
- RMSE: 3.1172 (RF) 
- RMSE: 3.0244 (GBM) 
- RMSE: 2.7618 (XGBoost) 
- RMSE: 2.7204 (LightGBM) 
- RMSE: 2.6482 (CatBoost)

# Hyperparameter Optimization
  
To enhance the performance of the machine learning models, hyperparameter optimization was carried out. This process involves tuning the parameters of the models to achieve the best possible performance. The technique used for hyperparameter optimization included:  

- Grid Search: An exhaustive search over a specified parameter grid.
      
Hyperparameter results for the top 3 Best-Performing Algorithms are as follows:  
- XGBoost  
RMSE (Before): 2.7618 (XGBoost)   
RMSE  (After): 2.7353 (XGBoost)   
XGBoost best params: {'learning_rate': 0.2, 'max_depth': None, 'n_estimators': 200}  

- LightGBM  
RMSE (Before): 2.7204 (LightGBM)   
RMSE  (After): 2.6919 (LightGBM)   
LightGBM best params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'n_estimators': 300}  

- CatBoost  
RMSE (Before): 2.6482 (CatBoost)   
RMSE  (After): 2.6259 (CatBoost)   
CatBoost best params: {'depth': 4, 'learning_rate': 0.1, 'n_estimators': 1500}  
# 5. Findings  
# Predictive Model Performance

Catboost and LightGBM models outperformed other models with the lowest RMSE and highest R-squared value. But, we opted for the LightGBM model due to its superior performance in terms of speed, memory efficiency, and high prediction accuracy, especially suitable for large-scale datasets and high-dimensional feature spaces.
  
# Visualizations
  
Below are some key visualizations from the analysis:  
  - Customer Satisfaction Distribution  
  - Feature Importance  
  - Model Performance Comparison  
 
# 6. Conclusion

The Interstellar Space Travel project successfully developed a predictive model for customer satisfaction in interstellar tourism. The findings provide valuable insights for enhancing service quality and customer experience. The LightGBM model, in particular, showed excellent performance in predicting satisfaction scores.
# 7. Future Work

Future work will focus on:

  - Expanding the dataset to include more diverse customer experiences.  
  - Improving the predictive models with advanced techniques such as deep learning.  
  - Integrating real-time feedback systems to continuously monitor and enhance customer satisfaction.  

# 8. Acknowledgements

We would like to thank all the participants who provided valuable feedback and data for this project. Special thanks to the development teams for their hard work and dedication. We also acknowledge Kaggle for providing the dataset used in this project.
# 9. Contact Information
  
- Gamze Öğütcü: https://www.linkedin.com/in/gamzeogutcu/  
- Eray Yumuşak: https://www.linkedin.com/in/eray-yumusak/  
- Yasin Akkök: https://www.linkedin.com/in/yasin-akk%C3%B6k-885515269/  

# Streamlit Application  
https://interstellarspacetravel.streamlit.app/
