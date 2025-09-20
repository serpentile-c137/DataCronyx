# Titanic Dataset EDA Summary

## 1. Dataset Overview and Structure

The Titanic dataset is a classic dataset often used for introductory machine learning and data analysis exercises.  It contains information about passengers aboard the RMS Titanic, including whether they survived the disaster.

**Dataset Source:** (Assuming a typical source) Kaggle, or similar online repository.

**Data Dictionary (Hypothetical):**

*   **PassengerId:**  Unique identifier for each passenger (Integer).
*   **Survived:**  Indicates whether the passenger survived (0 = No, 1 = Yes) (Integer).  *Target Variable*
*   **Pclass:** Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) (Integer).
*   **Name:** Passenger name (String).
*   **Sex:** Passenger gender (Male/Female) (String).
*   **Age:** Passenger age (Float).
*   **SibSp:** Number of siblings/spouses aboard (Integer).
*   **Parch:** Number of parents/children aboard (Integer).
*   **Ticket:** Ticket number (String).
*   **Fare:** Passenger fare (Float).
*   **Cabin:** Cabin number (String).
*   **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) (String).

**Shape:** (Assuming a standard shape for the training set) Approximately 891 rows (passengers) and 12 columns.

**Data Types:** The dataset consists of a mix of numerical (integer and float) and categorical (string) data types.  Proper type handling is crucial for analysis.

## 2. Data Quality Assessment

Based on typical observations of the Titanic dataset, the following data quality issues are commonly found:

*   **Missing Values:**
    *   **Age:**  A significant number of missing values are usually present in the `Age` column.
    *   **Cabin:**  The `Cabin` column typically has a large proportion of missing values.
    *   **Embarked:**  A small number of missing values may be present in the `Embarked` column.
*   **Data Type Inconsistencies:** None expected after initial load, but may need to be verified if the data was manually edited.
*   **Outliers:**
    *   **Fare:**  Outliers might be present in the `Fare` column, representing very expensive tickets.
    *   **Age:**  There might be some unrealistic age values (e.g., negative ages or extremely high ages).
*   **Duplicate Data:**  PassengerId should be unique, and duplicates should be investigated.  Duplicated rows (across all columns) would be highly unusual.
*   **Inconsistent Categorical Data:** The `Sex` and `Embarked` columns should have consistent values (e.g., "male" and "female", not "Male" and "Female").

## 3. Key Statistical Insights

*   **Survival Rate:**  The overall survival rate is typically around 38%.
*   **Age Distribution:**  The age distribution is skewed towards younger passengers, with a median age around 28-30 years.  A significant number of children were present.
*   **Fare Distribution:**  The fare distribution is highly skewed, indicating that most passengers paid relatively low fares, while a few paid very high fares.
*   **Passenger Class:**  The majority of passengers were in 3rd class.
*   **Gender Distribution:**  The dataset usually contains slightly more male than female passengers.
*   **Descriptive Statistics (Example):**

    | Feature  | Mean    | Std     | Min   | 25%   | 50%   | 75%   | Max   |
    | -------- | ------- | ------- | ----- | ----- | ----- | ----- | ----- |
    | Age      | ~29.7   | ~14.5   | 0.42  | 21.0  | 28.0  | 39.0  | 80.0  |
    | Fare     | ~32.2   | ~49.7   | 0.0   | 7.91  | 14.45 | 31.0  | 512.3 |
    | SibSp    | ~0.5    | ~1.1    | 0     | 0     | 0     | 1     | 8     |
    | Parch    | ~0.4    | ~0.8    | 0     | 0     | 0     | 0     | 6     |
    | Survived | ~0.38   | ~0.49   | 0     | 0     | 0     | 1     | 1     |

## 4. Patterns and Correlations Discovered

*   **Survival vs. Passenger Class:**  Passengers in 1st class had a significantly higher survival rate than those in 2nd and 3rd class.
*   **Survival vs. Gender:**  Females had a much higher survival rate than males.
*   **Survival vs. Age:**  Children had a higher survival rate.  Older passengers had a lower survival rate.
*   **Survival vs. SibSp/Parch:**  Passengers with few siblings/spouses or parents/children aboard had a slightly higher survival rate.  Passengers who were alone or had very large families tended to have lower survival rates.
*   **Survival vs. Fare:**  Passengers who paid higher fares had a higher survival rate.
*   **Survival vs. Embarked:**  Passengers who embarked from Cherbourg (C) had a slightly higher survival rate.
*   **Correlation Matrix (Hypothetical):**  A correlation matrix would reveal the linear relationships between numerical features.  For example, a moderate positive correlation might exist between `Fare` and `Pclass` (negative, as higher Pclass is lower class number), and a weak correlation between `Age` and `Survived`.

    *Example Correlation Matrix (Illustrative)*