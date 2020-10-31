# HOUSE PREDICTION PROJECT

# import modules
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# import warning and exceptions
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Read the train csv.
df = pd.read_csv('/house_price_prediction/data/raw/train.csv')
df.head()


# Read the csv with function.

def load_house_price():
    dataframe = pd.read_csv('/house_price_prediction/data/raw/train.csv')
    return dataframe


df = load_house_price()

# EDA
df.head()
df.shape
df.info()
df.isnull().values.any()

# Examine categorical variables
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical variable count: ', len(cat_cols))

# There are some categorical variables such as name contains too much classes so we need to exclude them

# How to calculate a categorical variable class ratio and target median effect?
pd.DataFrame({'Alley': df['Alley'].value_counts(),
              'ratio': df['Alley'].value_counts() / len(df),
              'target_median': df.groupby('Alley')['SalePrice'].median()})


# Lets apply it with a cat_summary function
def cat_summary(data, categorical_cols, target, number_of_classes=10):
    var_count = 0
    vars_more_classes = []
    for var in categorical_cols:
        if len(data[var].value_counts()) <= number_of_classes:
            print(pd.DataFrame({var: data[var].value_counts(),
                                'ratio': data[var].value_counts() / len(data),
                                'target_median': data.groupby(var)[target].median()}), end='\n\n\n')
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)

    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


# Call the function
cat_summary(df, cat_cols, 'SalePrice', 10)

# lets examine the variables which have more than 10 classes.
for col in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    print(df[col].value_counts())
    print(len(df[col].value_counts()))

# Do we need to exclude those? No.
# Call the function with change num_of_classes variable to 25
cat_summary(df, cat_cols, 'SalePrice', 25)

# Examine numeric variables

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in 'Id']
print('Numeric variable count: ', len(num_cols))
print(num_cols)

# For numeric variables we need to check distribution of values and outliers
df['LotArea'].hist(bins=20)
plt.show()


# Lets apply a function for all numeric variables
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


# Call the function
hist_for_nums(df, num_cols)

# Target analysis
df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# check all correlations
df[num_cols].corr()


# write a function to find highly correlated variables.
def find_correlation(dataframe, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low, high = find_correlation(df)


# Data processing and feature engineering

# Lets find variables contains rare classes

rare_cols = [col for col in df.columns if len(df[col].value_counts()) <= 25
                and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


# where are 47 columns contains rare classes.
len(rare_cols)

# Lets create a function to find all rare class contained columns

def rare_analyser(dataframe, target, rare_perc=0.01):

    rare_columns = [col for col in df.columns if len(df[col].value_counts()) <= 20
                    and df[col].dtypes == 'O'
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")

# call the function
rare_analyser(df, "SalePrice")


# Lets create a function collect all rare classes under 'Rare' class for categorical variables
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# Call the function
df = rare_encoder(df, 0.01)


# examine rare situations and select redundant variables.
rare_analyser(df, 'SalePrice', 0.01)
drop_list = ['MiscFeature', 'PoolQC', 'LandSlope', 'Utilities', 'Street']


# update cat_cols list
cat_cols = [col for col in df.columns if df[col].dtypes == 'O'
            and col not in drop_list]


# drop drop list columns from dataset
for col in drop_list:
    df.drop(col, axis=1, inplace=True)

cat_summary(df, cat_cols, "SalePrice")


# LABEL ENCODING & ONE HOT ENCODING

# Lets define a function to create one_hot_encoding columns for categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns

df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()
new_cols_ohe
len(new_cols_ohe) # 188 new variables created.

cat_summary(df, new_cols_ohe, "SalePrice")

# Missing values

# Function to find columns with null values
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na

# fill all null values with median of the variable
df = df.apply(lambda x: x.fillna(x.median()), axis=0)
missing_values_table(df)


# outliers

# find lower and upper limits
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# find columns with outlier counts
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names

# call the function
has_outliers(df, num_cols)


# Lets apply suppression to outliers.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# apply for all columns
for col in num_cols:
    replace_with_thresholds(df, col)

# Control outliers after suppression.
has_outliers(df, num_cols)

# Standardization
# It can be applied only numeric columns
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]

df[cols_need_scale].head()
df[cols_need_scale].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
hist_for_nums(df, cols_need_scale)


def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


for col in cols_need_scale:
    df[col] = robust_scaler(df[col])

# Modelleme

X = df.drop('SalePrice', axis=1)
y = df[["SalePrice"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)



