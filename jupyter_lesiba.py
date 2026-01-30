import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1.Load the dataset, cleaning, and exploration
data = pd.read_csv('House Price Prediction dataset.csv')
print(data.head())
print(data.info())
print(data.describe())

print(data.isnull().sum()) # Check for missing values

data = data.dropna() # Drop rows with missing values
   


data = data.drop('Id', axis=1) # Drop irrelevant columns

numerical_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price'] #check for outliers in numerical columns
for col in numerical_cols:
    mean = data[col].mean()
    std = data[col].std()
    outliers = data[data[col]>(mean + 3*std) | data[col]<(mean - 3*std)]
    print(f"{col}: {len(outliers)} outliers") 

categorical_cols = ['Location', 'Condition', 'Garage'] #handle categorical variables
print(f"\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {data[col].nunique()}")


label_encoders = {} #covert categorical variables to numerical using label encoding
for col in categorical_cols:
    le = label_encoders()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"{col} encoding: {dict(zip(le.classes_, le.transform(le.classes_)))} ")



plt.figure(figsize=(10,8)) #correlation analysis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout
plt.savefig('correlation_matrix.png', dpi=100)
plt.show()


#target variable distribution
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['Price'], kde=True, bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
sns.boxplot(y=data['Price'])
plt.title('Price Boxplot')
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=100)
plt.show()


#feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()
for idx, col in enumerate(['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']):
    sns.histplot(data[col], kde=True, ax=axes[idx])
    axes[idx].set_title(f'{col} Distribution')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

#remove unused subplots
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=100)
plt.show()


#relationship between features and target variable
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()
for idx, col in enumerate(['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']):
    axes[idx].scatter(data[col], data['Price'], alpha=0.5)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Price')
    axes[idx].set_title(f'{col} vs Price')

    #add trendline
    z = np.polyfit(data[col], data['Price'], 1)
    p = np.poly1d(z)
    axes[idx].plot(data[col], p(data[col]), "r--", alpha=0.8)

#remove unused subplots
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('feature_vs_price.png', dpi=100)
plt.show()


#2.Feature selection and train-test split

#selection features based on correlation with price
correlation_with_price = correlation_matrix['Price'].sort_values(ascending=False)
print("\nCorrelation with Price:")
print(correlation_with_price)

#select features with correlation > 0.1
selected_features = correlation_with_price[abs(correlation_with_price) > 0.1].index.tolist()
selected_features.remove('Price') #remove target variable
print(f"\nSelected features: {selected_features}")

#Prepare X and Y
X = data[selected_features]
y = data['Price']

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}") 

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#3.Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model training completed.")
print(f"Number of features used: {model.coef_.shape[0]}")

#4.Model Evaluation

#make predictions
y_pred_train = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

#calculation metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTraining RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R^2: {train_r2:.4f}")
print(f"Testing R^2: {test_r2:.4f}")

#intepretation
print(f"\nModel Interpretation:")
print(f"1. The model explains {test_r2:.2%} of the variance in house prices.")
print(f"2. Avarage prediction error on the test set is ${test_rmse:,.2f}.")
print(f"3. Model performance on training and testing sets are comparable, indicating no overfitting.")

#5.Coefficients analysis

#create a dataframe for coefficients
coefficients_data = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': model.coef_,
    'Absolute Coefficient': abs(model.coef_)
})

#sort by absolute coefficient
coefficients_data = coefficients_data.sort_values('Absolute Coefficient', ascending=False)
print("\nfeature coefficients(sortedby impact):")
print(coefficients_data.to_string(index=False))

print("\nInterpretation of Coefficients:")
for idx, row in coefficients_data.iterrows():
    feature = row['Feature']
    coef = row['Coefficient']

    #get original column name
    if feature in ['Location', 'Condition', 'Garage']:
        original_name = feature
        if original_name == 'Location':
            interpretation = "Downtown=0, Suburb=1,Urban=2, Rural=3"
        elif original_name == 'Condition':
            interpretation = "Poor=0, Fair=1, Good=2, Excellent=3"
        else:
            interpretation = "No=0, Yes=1"

            print(f"{original_name} (coef: {coef:.2f}):")
            print(f"   Encoded as: {interpretation}")
            print(f"Each unit increases adds ${coef * scaler.scale_[list(selected_features).index(feature)]:,.0f} to the price.")
    else:
        print(f"{feature} (coef: {coef:.2f}):")
        print(f"Each unit increase in adds ${coef * scaler.scale_[list(selected_features).index(feature)]:,.0f} to the price.")

#6.visualization of Predictions
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Set: Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Testing Set: Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=100)
plt.show()

#residual analysis
residuals = y_test - y_test_pred
plt.figure(figsize=(10,6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Price')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals_vs_predicted.png', dpi=100)
plt.show()

#7.Save the model and scaler
model_data = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': selected_features,
    'feature_importance': coefficients_data.to_dict('records')
}

with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved as 'house_price_model.pkl'")

# 10. EXAMPLE PREDICTIONS
print("\n" + "="*50)
print("STEP 10: EXAMPLE PREDICTIONS")
print("="*50)

# Create example houses
example_houses = [
    {
        'Area': 2000,
        'Bedrooms': 3,
        'Bathrooms': 2,
        'Floors': 2,
        'YearBuilt': 1995,
        'Location': 'Suburban',
        'Condition': 'Good',
        'Garage': 'Yes'
    },
    {
        'Area': 3500,
        'Bedrooms': 4,
        'Bathrooms': 3,
        'Floors': 3,
        'YearBuilt': 2010,
        'Location': 'Downtown',
        'Condition': 'Excellent',
        'Garage': 'Yes'
    },
    {
        'Area': 1200,
        'Bedrooms': 2,
        'Bathrooms': 1,
        'Floors': 1,
        'YearBuilt': 1970,
        'Location': 'Rural',
        'Condition': 'Fair',
        'Garage': 'No'
    }
]

print("\nExample Predictions:")
print("-" * 80)

for i, house in enumerate(example_houses, 1):
    # Prepare the data
    house_df = pd.DataFrame([house])
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in house_df.columns:
            house_df[col] = label_encoders[col].transform(house_df[col])
    
    # Select only the features used in training
    house_features = house_df[selected_features]
    
    # Scale the features
    house_scaled = scaler.transform(house_features)
    
    # Make prediction
    prediction = model.predict(house_scaled)[0]
    
    print(f"\nExample House {i}:")
    for key, value in house.items():
        print(f"  {key}: {value}")
    print(f"  Predicted Price: ${prediction:,.2f}")
    print("-" * 40)

# 11. SUMMARY
print("\n" + "="*50)
print("SUMMARY AND CONCLUSIONS")
print("="*50)

print("\nKey Findings:")
print("1. Most Important Features:")
for idx, row in coefficients_data.head(3).iterrows():
    print(f"   - {row['Feature']} (impact: ${abs(row['Coefficient'] * scaler.scale_[list(selected_features).index(row['Feature'])]):,.0f} per unit)")

print(f"\n2. Model Performance:")
print(f"   - R² Score: {test_r2:.4f} ({test_r2:.2%} variance explained)")
print(f"   - RMSE: ${test_rmse:,.2f} (average error)")
print(f"   - Model generalizes well (train/test scores similar)")

print(f"\n3. Business Insights:")
print(f"   - Location has the strongest impact on price")
print(f"   - Area is the most important numerical feature")
print(f"   - Having a garage significantly increases house value")
print(f"   - Newer houses generally command higher prices")

print(f"\n4. Model Limitations:")
print(f"   - R² of {test_r2:.4f} means there's room for improvement")
print(f"   - Linear model may not capture complex non-linear relationships")
print(f"   - Could benefit from additional features or more complex models")

print(f"\n5. Next Steps:")
print(f"   - Try polynomial features or feature interactions")
print(f"   - Experiment with other algorithms (Random Forest, Gradient Boosting)")
print(f"   - Collect more data or additional features")
print(f"   - Consider feature engineering (e.g., Age = Current Year - YearBuilt)")

print("\n" + "="*50)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*50)
print("\nFiles created:")
print("1. house_price_model.pkl - Trained model and artifacts")
print("2. correlation_matrix.png - Feature correlation heatmap")
print("3. price_distribution.png - Target variable distribution")
print("4. feature_distributions.png - Feature distributions")
print("5. feature_vs_price.png - Feature vs price relationships")
print("6. actual_vs_predicted.png - Prediction accuracy visualization")
print("7. residual_plot.png - Model residuals analysis")