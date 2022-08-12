# Seoul Bike Sharing Demand Prediction

**AlmaBetter Verfied Project** 
![image](https://user-images.githubusercontent.com/91868265/184426395-a5c9fea2-0150-4866-a29f-683e52a4698e.png)




**Abstract**

As more number of rented bikes are being used in the cities nowadays, it becomes important for the company to predict the number of required rental bikes required across a day so that no demand supply gap would be generated for rental bikes. This project aims at providing necessary solution to predict the rental bikes demand using machine learning algorithms so that all the stakeholders of the business can be satisfied.


# 📋 **Problem Statement**
Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The cru cial part is the prediction of bike count required at each hour for the stable supply of rental bikes.

## **Data Description**
•	Date : year-month-day
•	Rented Bike count - Count of bikes rented at each hour
•	Hour - Hour of the day
•	Temperature-Temperature in Celsius
•	Humidity - %
•	Wind speed - m/s
•	Visibility - 10m
•	Dew point temperature - Celsius
•	Solar radiation - MJ/m2
•	Rainfall - mm
•	Snowfall - cm
•	Seasons - Winter, Spring, Summer, Autumn
•	Holiday - Holiday/No holiday
•	Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)


Introduction
The present scenario is about how good is the customer service in any industry as the number of options at the customer’s disposal is unlimited. So, it becomes extremely important to make sure that the customers will not be made to wait for the rental bikes. It would also not be practical to keep lot of bikes even when the demand is low. Hence, with the help of machine learning, this project aims at predicting the rental bike demand so that no problems arise.

## **Data Pipeline**

	Exploratory Data Analysis:
The first step of our project is performing the EDA process on the dataset so that we can get the idea about the dataset i.e. the number of variables, the data type of the variables , visualize the dataset for Better understanding and decide the suitable methods and algorithms that might produce desired outcomes

	Data Preprocessing:
In EDA process we find the type of dataset and decide the approach, in this project the preprocessing steps would removing the punctuations, stopwords , generate count vectorizer and document term matrix which would help in building up the model.

	Building Machine Learning Model:
After the data preprocessing is done then the data will be ready to be fit into machine learning models .For current problem statement topic modeling approach would be suitable . In topic modeling, a topic is defined by a cluster of words with each word in the cluster having a probability of occurrence for the given topic, and different topics have their respective clusters of words along with corresponding probabilities.

## 💾 **Project Files Description**

This Project includes 1 colab notebook and 1 Pdf of presentation.

### **Executable Files:**

[Bike Sharing Demand Prediction]-(https://github.com/anas084464/Bike-Sharing-Demand-Prediction/blob/main/Colab_Bike_Sharing_Demand_Prediction_Capstone_Project.ipynb)- Includes Exploratory Data Analysis and all algorithms which are used in this project.

[Bike Sharing Demand prediction.pdf]-(https://github.com/anas084464/Bike-Sharing-Demand-Prediction/blob/main/presentation%20-%20Seoul%20bike%20sharing%20demand%20prediction.pptx)- Includes pdf of the presentation of the project.

### **Output:**

[Google Colab]-(https://github.com/anas084464/Bike-Sharing-Demand-Prediction/blob/main/Colab_Bike_Sharing_Demand_Prediction_Capstone_Project.ipynb)

## 📋 **Execution Instruction**

The order of execution of the colab notebook is as follows:

**1) Bike_Sharing_Demand_Prediction_Capstone_Project.ipynb**

First, click on the open in colab button present on the top centre of the notebook.

In this .ipynb file, we have -

• EDA on Bike sharing demand prediction.

• Hypothesis.

• Fitting different models and cross validate them.

**2) Kaggle Dataset**

Downlaod the dataset from kaggle through provided link.Then, connect to the runtime and execute the cell to mount the drive or upload the data file to the current runtime.

**3) Cell Path**

Finally, delete the path in the dataset loading cell and replace it with the path of your current data file. Run each cell to see the output below it.

**Algorithms Used**

1. Linear Regression
2. Lasso and Ridge Regression
3. Decision tree
4. Random Forest
5. Gradient Boosting
       
## **Conclusions**
The project comes to an end at this point. Beginning with loading the dataset, so far we have done EDA, pre-processing the data, Label encoding, Scaling the data, splitting the data into train and test data, applying various machine learning algorithms followed by hyper parameter tuning. We implemented 8 M.L. models. After comparing the mean square error and  mean root square error of all the models, XGBoost has least mean square  error and root mean square error. XGBoost has highest accuracy of 91.9%  among all algorithms. So, We can conclude that XGBoost is the best model  to predict rented bike count. The number of business hours of the day and the demand for rented bikes  were most correlated and It makes sense also. Highest number of  bike  rented at the 18th hour of day. Total number of bike count increased when there was favourable  temperature. So, this can be an important factor in predicting underlying  patterns of rented bike count.


## 📜 **Credits**

Anas Mustafa | Chetan Rajput | Sarthak Rastogi

## 📚  **References**

Random Forest Regressor - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
Gradient Boosting Documentation - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

