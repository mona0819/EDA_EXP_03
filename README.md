**EXP 3 - Delhi Air Quality Analysis**
```
Name: Mohana Priya D
Reg No.: 212225230182
```
**Aim**


To compare air quality parameters in Delhi across different stations and analyze the relationship between pollutants (e.g., PM2.5 and NO₂) using scatter plots and correlation analysis.


**Procedure / Algorithm**

1)Load the dataset using pandas.

2)Preprocess the data:

3)Convert the date column (period.datetimeFrom.utc) to datetime format.

4)Drop missing or invalid values.

5)Pivot the dataset so each pollutant (parameter) becomes a separate column.

6)Plot scatter plot between PM2.5 and NO₂ to study their relationship.

7)Plot correlation heatmap between all pollutants to identify relationships.

8)Interpret the results — identify which pollutants are correlated and which stations are most polluted.


**Program**

**Name : Mohana Priya D**

**Reg No: 212225230182**

#Q1. Load the dataset and perform preprocessing
```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("del-sirifort-cpcb-2024-25.csv")

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Select required columns
columns = [
    'PM2.5 (µg/m³)',
    'PM10 (µg/m³)',
    'NO2 (µg/m³)',
    'Ozone (µg/m³)'
]

data = df[columns].copy()

# Convert selected columns into numeric format
for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Check missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Handle missing values separately for each variable
clean_data = data.copy()

print("\nAvailable Records After Preprocessing:")
for col in columns:
    print(col, ":", clean_data[col].count())

print("\nDataset Information:")
print(clean_data.info())

print("\nPreprocessed Dataset:")
print(clean_data.head())
```
#Q2. Distribution analysis using histograms
```python
plt.figure(figsize=(14, 10))

for i, col in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    sns.histplot(
        clean_data[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel("Concentration (µg/m³)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```
#Q3. Calculate numerical summary
```python
summary = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()

    summary.loc[col, 'Mean'] = x.mean()

    summary.loc[col, 'Median'] = x.median()

    summary.loc[col, 'Mode'] = x.mode().iloc[0]

    summary.loc[col, 'Minimum'] = x.min()

    summary.loc[col, 'Q1'] = x.quantile(0.25)

    summary.loc[col, 'Q3'] = x.quantile(0.75)

    summary.loc[col, 'Maximum'] = x.max()

print("NUMERICAL SUMMARY")
print(summary.round(2))
```
#Q4. Calculate measures of spread
```python
spread = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()

    mean = x.mean()

    # Range
    spread.loc[col, 'Range'] = x.max() - x.min()

    # Variance
    spread.loc[col, 'Variance'] = x.var()

    # Standard Deviation
    spread.loc[col, 'Standard Deviation'] = x.std()

    # Coefficient of Variation
    spread.loc[col, 'Coefficient of Variation (%)'] = (
        (x.std() / mean) * 100 if mean != 0 else np.nan
    )

print("MEASURES OF SPREAD")
print(spread.round(2))
```
#Q5. Construct box plots and detect outliers using IQR
```python
#Box plot visualization
plt.figure(figsize=(12, 7))

sns.boxplot(data=clean_data)

plt.title("Box Plot of Delhi Air Quality Pollutants")
plt.xlabel("Air Pollutants")
plt.ylabel("Concentration (µg/m³)")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```
#Outlier detection using IQR
```python
for col in columns:

    x = clean_data[col].dropna()

    # Calculate quartiles
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = x[
        (x < lower_limit) | (x > upper_limit)
    ]

    print("\nPollutant:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))

    print("First 10 Outliers:")
    print(outliers.head(10).values)
```
#Q6. Perform Min-Max scaling
```python
# Create an empty DataFrame
minmax_data = pd.DataFrame(index=clean_data.index)

for col in columns:

    x = clean_data[col]

    min_value = x.min()
    max_value = x.max()

    # Apply Min-Max scaling
    if max_value != min_value:
        minmax_data[col] = (
            (x - min_value) / (max_value - min_value)
        )
    else:
        minmax_data[col] = 0.0

print("MIN-MAX SCALED DATA")
print(minmax_data.head(10))

# Display minimum and maximum after scaling
print("\nScaled Data Range:")
print(minmax_data.agg(['min', 'max']).round(4))

# Save complete scaled dataset
minmax_data.to_csv("minmax_scaled_data.csv", index=False)
```
#Q7. Perform Z-score standardization using StandardScaler
```python
# Create an empty DataFrame
standardized_data = pd.DataFrame(index=clean_data.index)

# Apply StandardScaler separately to each pollutant
for col in columns:

    valid = clean_data[col].notna()

    scaler = StandardScaler()

    standardized_data.loc[valid, col] = (
        scaler.fit_transform(
            clean_data.loc[valid, [col]]
        ).flatten()
    )

print("Z-SCORE STANDARDIZED DATA")
print(standardized_data.head(10))

# Verify mean and standard deviation
print("\nMean After Standardization:")
print(standardized_data.mean().round(4))

print("\nStandard Deviation After Standardization:")
print(standardized_data.std(ddof=0).round(4))

# Save complete standardized dataset
standardized_data.to_csv(
    "zscore_standardized_data.csv",
    index=False
)
```
#Q8. Calculate Gini coefficient
```python
# Function to calculate Gini coefficient
def gini_coefficient(values):

    # Convert values into NumPy array
    x = np.array(values, dtype=float)

    # Remove missing values
    x = x[~np.isnan(x)]

    # Gini requires non-negative values
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values")

    # Handle empty or zero-valued data
    if len(x) == 0:
        return np.nan

    if np.sum(x) == 0:
        return 0

    # Sort values in ascending order
    x = np.sort(x)

    n = len(x)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)

    gini = (
        2 * np.sum(index * x) / (n * np.sum(x))
    ) - ((n + 1) / n)

    return gini


# Calculate Gini for all four pollutants
print("GINI COEFFICIENT OF AIR POLLUTANTS")

for col in columns:

    gini = gini_coefficient(clean_data[col])

    print(col, ":", round(gini, 4))
```
```python
# Save numerical summary
summary.to_csv("numerical_summary.csv")

# Save measures of spread
spread.to_csv("measures_of_spread.csv")

print("All analysis results saved successfully!")
```
**Output**
<img width="1201" height="782" alt="image" src="https://github.com/user-attachments/assets/e24e5a22-4570-42e0-8f3d-43b079a7b17f" />
<img width="803" height="592" alt="image" src="https://github.com/user-attachments/assets/037f228a-e018-437c-b06f-6019a64edc35" />
<img width="750" height="191" alt="image" src="https://github.com/user-attachments/assets/d1dedb91-7d7a-49d6-a3e5-c51f1db2c130" />
<img width="707" height="330" alt="image" src="https://github.com/user-attachments/assets/8ec6a393-9669-44df-972b-19d40584e9c7" />
<img width="778" height="466" alt="image" src="https://github.com/user-attachments/assets/5e078a22-3196-49ea-b66e-f1b17d68433c" />
<img width="847" height="758" alt="image" src="https://github.com/user-attachments/assets/3225eaa6-313c-4938-9650-d31d6c1a1c94" />
<img width="761" height="462" alt="image" src="https://github.com/user-attachments/assets/712d18b9-91de-40f7-9d4a-0d98ca287048" />
<img width="733" height="670" alt="image" src="https://github.com/user-attachments/assets/9f4e53c6-9f38-4c1c-b887-3e9b4228bf7d" />
<img width="482" height="172" alt="image" src="https://github.com/user-attachments/assets/53e0ce1e-0b22-4c96-b6ed-403c7e5fbf54" />
<img width="532" height="120" alt="image" src="https://github.com/user-attachments/assets/5da7da6c-dd7f-4280-8c6b-318ae9a424fc" />


**Interpretation**

1. PM2.5 and NO₂ show a strong positive correlation, suggesting that both pollutants increase together, possibly due to common sources such as vehicle and industrial emissions.

2. The histograms show the distribution of different pollutants. The pollutant concentrations are not uniformly distributed, indicating variations in air pollution levels.

3. The numerical summary shows the mean, median, mode, minimum, maximum, and quartile values of each pollutant, helping to understand their concentration levels.

4. The measures of spread indicate variations in pollutant concentrations. Standard deviation and variance help identify how much the pollution levels differ from their average values.

5. The box plots show the presence of outliers in pollutant concentrations, indicating certain observations with unusually high pollution levels.

6. The IQR method successfully identifies extreme values in PM2.5, PM10, NO₂, and Ozone concentrations.

7. Min-Max scaling transforms pollutant values into a common range between 0 and 1, making it easier to compare different pollutants.

8. Z-score standardization converts pollutant values into a standardized scale with a mean approximately equal to 0 and standard deviation approximately equal to 1.

9. The Gini coefficient measures the inequality in pollutant concentration values. A higher Gini coefficient indicates greater variation in the distribution of pollution levels.

10. The analysis helps understand the distribution, variability, outliers, and relative concentration patterns of different air pollutants in Delhi.

**Result**

The dataset was successfully loaded and processed to extract pollutant-wise and station-wise air quality data for Delhi.


