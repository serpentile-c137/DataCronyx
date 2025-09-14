```markdown
# Titanic Dataset EDA Summary

This document summarizes the Exploratory Data Analysis (EDA) performed on the Titanic dataset.

## Dataset Overview

- **Shape:** (891, 12)
- The dataset contains 891 rows and 12 columns.
- **Columns:** `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

## Data Types

```
PassengerId      int64
Survived         int64
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object
```

## Summary Statistics

```
              PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count         891.000000  891.000000  891.000000  891.000000  891.000000  891.000000  891.000000
mean          446.000000    0.383838    2.308642   29.361582    0.523008    0.381594   32.204208
std           257.353842    0.486592    0.836071   13.019697    1.102743    0.806057   49.693429
min             1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%           223.500000    0.000000    2.000000   22.000000    0.000000    0.000000    7.910400
50%           446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%           668.500000    1.000000    3.000000   35.000000    1.000000    0.000000   31.000000
max           891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
```

## Missing Values

### Initial Missing Values:

```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

### Handling Missing Values:

- **Age:** Missing values imputed with the median age (28.0).
- **Cabin:** Dropped the `Cabin` column due to a large number of missing values.
- **Embarked:** Missing values imputed with the mode ('S').

### Missing Values After Imputation:

```
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Embarked       0
dtype: int64
```

## Outlier Detection (Boxplots)

Boxplots were generated for numerical features (`Age`, `Fare`, `SibSp`, `Parch`) to identify potential outliers. The plots are saved as `/code/boxplots.png`.

## Distribution of Numerical Features (Histograms)

Histograms were plotted for numerical features (`Age`, `Fare`, `SibSp`, `Parch`) to visualize their distributions. Kernel Density Estimation (KDE) was also included. The plots are saved as `/code/histograms.png`.

## Count Plots of Categorical Features

Count plots were created for categorical features (`Survived`, `Pclass`, `Sex`, `Embarked`) to show the distribution of each category. The plots are saved as `/code/countplots.png`.

## Survival Analysis

### Survival Rate by Sex:

```
Sex
female    0.742038
male      0.188908
Name: Survived, dtype: float64
```

- Females had a significantly higher survival rate (74.2%) compared to males (18.9%).

### Survival Rate by Pclass:

```
Pclass
1    0.629630
2    0.472826
3    0.242363
Name: Survived, dtype: float64
```

- Passengers in Pclass 1 had the highest survival rate (62.9%), followed by Pclass 2 (47.3%), and Pclass 3 had the lowest survival rate (24.2%).
```
