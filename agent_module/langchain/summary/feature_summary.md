## Feature Engineering Summary - Titanic Dataset

This document summarizes the feature engineering process applied to the Titanic dataset (`../example_dataset/titanic.csv`).

### 1. New Features Created

The following new features were engineered to potentially improve model performance:

*   **Title:** Extracted the title from the 'Name' feature (e.g., Mr., Mrs., Miss., Master.). This captures social status and potentially survival likelihood.
*   **FamilySize:** Combined 'SibSp' (siblings/spouses aboard) and 'Parch' (parents/children aboard) to represent the total family size.
*   **IsAlone:** Binary feature indicating whether the passenger was traveling alone (FamilySize = 1).
*   **Age_imputed:** Imputed missing 'Age' values using the median age, grouped by 'Sex' and 'Pclass'. This reduces data loss and potential bias from removing rows with missing age.
*   **Fare_Per_Person:** Calculated fare per person by dividing 'Fare' by 'FamilySize' (or 1 if FamilySize is 0). This could provide a more granular view of the cost of travel.
*   **Cabin_Deck:** Extracted the first letter of the 'Cabin' feature to represent the deck level. Missing cabin values were assigned a 'Unknown' deck.
*   **Embarked_imputed:** Imputed missing 'Embarked' values with the most frequent value (mode).

### 2. Feature Selection Rationale

The following rationale guided feature selection:

*   **Relevance:** Selecting features directly related to survival (e.g., age, class, sex, family size, fare).
*   **Reducing Redundancy:** Combining related features (SibSp, Parch -> FamilySize).
*   **Addressing Missing Values:** Imputing missing values to retain data and avoid bias.
*   **Domain Knowledge:** Leveraging domain knowledge (e.g., title reflecting social status, cabin deck reflecting location on the ship).
*   **One-Hot Encoding:** Categorical features like 'Sex', 'Embarked', 'Title', and 'Cabin_Deck' were one-hot encoded to be used in machine learning models.  'Pclass' was also one-hot encoded, as while it is ordinal, the relationship between classes may not be linear.

Features like 'Name', 'Ticket', and the original 'SibSp', 'Parch', and 'Fare' features were dropped after the creation of new features. 'Age' and 'Embarked' were dropped after imputation and one-hot encoding.  'Cabin' was dropped after the extraction of 'Cabin_Deck'.

### 3. Expected Impact on Model Performance

*   **Improved Accuracy:** The new features aim to capture more nuanced relationships between passenger characteristics and survival, leading to better model accuracy.
*   **Reduced Bias:** Imputing missing values and handling categorical variables appropriately can reduce bias in the model.
*   **Enhanced Generalization:** By engineering features that generalize well to unseen data, the model should perform better on new passengers.
*   **Better Feature Importance Understanding:** The feature engineering process will allow for a better understanding of the factors that influenced survival on the Titanic.

### 4. Feature Importance Insights

While specific feature importance insights would come from analyzing the trained model, we anticipate the following:

*   **Sex:** Likely to be a strong predictor, with females having a higher survival rate.
*   **Pclass:** Passenger class will likely be important, with higher classes having a higher survival rate.
*   **Title:** Titles such as 'Mr.' may have a lower survival rate, whereas 'Mrs.' and 'Miss.' may have a higher survival rate. 'Master' may indicate a higher survival rate for young boys.
*   **FamilySize and IsAlone:** Family size and whether a passenger was alone could influence survival, with larger families potentially having a lower chance of survival.
*   **Age:** Age is expected to be a relevant factor, with younger passengers potentially having a higher survival rate.
*   **Fare_Per_Person:** Fare per person may provide a more nuanced view of the cost of travel and its relationship to survival.
*   **Cabin_Deck:** Cabin deck might be correlated with survival, potentially reflecting proximity to lifeboats.
*   **Embarked:** Embarkation port may have an influence, potentially due to different class distributions and lifeboat availability.

These are just initial expectations. The actual feature importances will be determined by the specific model used and the data itself. Model-specific analysis of feature importances (e.g., using coefficients from logistic regression or feature importance scores from tree-based models) would provide more concrete insights.