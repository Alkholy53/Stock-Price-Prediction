# Stock Price Prediction

This project investigates several machine learning models for predicting the closing price of the Tesla stock (TSLA) using historical data. The data is retrieved from Yahoo Finance and includes Open, High, Low, Close, Adj Close, and Volume for each day.

## Libraries Used

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- matplotlib.pyplot: Data visualization
- sklearn: Machine learning library
- tensorflow: Deep learning library

## Data Analysis

- **Data Gathering:** The dataset is retrieved from Yahoo Finance and includes daily data for Tesla's stock price from February 1st, 2018, to December 30th, 2022.
  
- **Data Exploration:** The data is explored to understand the distribution of features and identify potential relationships between them. Univariate and bivariate visualizations are created to gain insights.

- **Data Preprocessing:** The data is preprocessed by:
  - Dropping the Volume feature due to its lack of correlation with other features.
  - Splitting the data into training and testing sets for model evaluation.

## Machine Learning Models

Several machine learning models are evaluated for predicting the closing price of the Tesla stock:

- **Random Forest Regression:** This model uses a random forest ensemble to make predictions. Grid search is employed to find the optimal hyperparameters for the model.

- **AdaBoost Regression with Random Forest:** This model utilizes AdaBoost with a random forest regressor as the base estimator. The hyperparameters from the Random Forest grid search are used for AdaBoost.

- **Support Vector Regression (SVR):** This model uses a support vector machine for regression. Grid search is employed to find the optimal hyperparameters for the SVR model.

- **Elastic Net Regression:** This model uses a regularized linear regression technique with both L1 and L2 penalties for feature selection and shrinkage. Grid search is conducted to find the optimal hyperparameters for the Elastic Net model.

## Model Evaluation

Each model is evaluated using Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) on the testing set. The performance of each model is compared, and the model with the lowest MAPE is considered the best performing model.

## Long Short-Term Memory (LSTM) Model

An LSTM model is also implemented to explore deep learning techniques for stock price prediction. The model is trained on historical data to predict future closing prices. The performance of the LSTM model is compared to the machine learning models.

## Results

The project evaluates the performance of several machine learning models for predicting the closing price of the Tesla stock. The Random Forest Regression model achieved the lowest MAPE, indicating its effectiveness in this specific case. The LSTM model also showed promising results, suggesting the potential of deep learning for stock price prediction.

## Code Structure

The code for this project is organized into several Python scripts:

- **data_analysis.py:** Performs data exploration and visualization.
- **preprocessing.py:** Preprocesses the data for machine learning models.
- **machine_learning_models.py:** Implements and evaluates the machine learning models.
- **lstm_model.py:** Implements and evaluates the LSTM model.
