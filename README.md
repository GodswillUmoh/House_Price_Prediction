# House_Price_Prediction
__This Repo is to predict the House price in different Areas in USA__

---

## About The Dataset:
- The dataset was got from kaggle.com entitle: usa_house_price_prediction.

- The dataset is a csv file

---

### The Dataset
_Below are the first five row of the dataset_

| date| price| bedrooms| bathrooms | sqft_living | sqft_lot | floors | waterfront| view| condition|sqft_above|sqft_basement|yr_built|yr_renovated|street|city|statezip|country|
|---------|------|---------|-----------|-------------|-----------|-------|-----------|-----|--------|--------|---------|--------|--------|----------|--------|-------|------|
|5/2/2014| 0:00|313000|3|1.5|1340|7912|1.5|0|0|3|1340|0|1955|2005|18810 |Densmore Ave N|Shoreline|WA| 98133|USA|


## Python Code for the LinearRegression

### Load the dataset
```python
data = pd.read_csv('usa_house_price_pred.csv')
```

#### Display the first few rows
```python
print(data.head())
```

#### Check the summary statistics
```python
print(data.describe())
```

#### Data preprocessing
```python
data = pd.read_csv('usa_house_price_pred.csv')
# Select the relevant columns
data_2 = data[['bedrooms', 'price', 'sqft_living', 'sqft_lot']]

# Check for missing values
print(data_2.isnull().sum())
```

#### <h2> Split Data into Training and Testing Sets</h2>
```python
#The needed library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #pip install scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define the independent variable (X) and dependent variable (y)
X = data[['bedrooms']]
y = data['price']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### <h2>Create and Train the Model</h2>
```python
# Create the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)
```

#### <h2>Make Predictions</h2>
```python
 #Create the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

#summary, load and inspect data, select the x and y variables,
#split to train and test for both x and y = train_test_split(x, y, testsize=0.2)
#initialize model = the model (e.g LinearRegression)
#fit x and y train into model
#make prediction with y_prediction = model.predict(X_test)
```

#### Evaluate the Model<
```python
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Print the model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
```
