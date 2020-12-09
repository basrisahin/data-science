# HOUSE PREDICTION PROJECT

# Import dependencies
import pandas as pd
import numpy as np
import pymysql as pymysql
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# combine load and fix data in a one function.
def load_fix_house_price(class_count=20):
    train = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/train.csv")
    test = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/test.csv")
    data = train.append(test).reset_index()

    cat_seen_as_numeric = [col for col in data.columns if data[col].dtypes != 'O'
                           and len(data[col].value_counts()) < class_count]

    for col in cat_seen_as_numeric:
        data[col] = data[col].astype(object)

    return data


df = load_fix_house_price()

# Examine categorical variables
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Categorical variable count: ', len(cat_cols))


# Lets apply it with a cat_summary function
def cat_summary(data, categorical_cols, target, number_of_classes=20):
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


cat_summary(df, cat_cols, 'SalePrice')


# Examine numeric variables
num_cols = [col for col in df.columns if df[col].dtypes != 'O'
            and col not in 'Id'
            and col not in 'SalePrice']
print('Numeric variable count: ', len(num_cols))


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

# Data processing and feature engineering

# Check null values for each column
df.info()
# null values are extremely much.
near_null_columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

# fill null values with none, now it keep some info in it.
for col in near_null_columns:
    df[col] = df[col].fillna('None')


# Rare analyzer
def rare_analyser(dataframe, categorical_columns, target, rare_percentage=0.01):
    rare_columns = [col for col in categorical_columns if
                    (df[col].value_counts() / len(df) < rare_percentage).any(axis=None)]

    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


# call the function
rare_analyser(df, cat_cols, "SalePrice")


# Lets create a function collect all rare classes under 'Rare' class for categorical variables
def rare_encoder(dataframe, categorical_columns, rare_percentage=0.01):
    rare_columns = [col for col in categorical_columns
                    if (dataframe[col].value_counts() / len(dataframe) < rare_percentage).any(axis=None)]

    for var in rare_columns:
        tmp = dataframe[var].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_percentage].index
        dataframe[var] = np.where(dataframe[var].isin(rare_labels), 'Rare', dataframe[var])

    return dataframe


# Call the function
df = rare_encoder(df, cat_cols, 0.01)

# examine rare situations and select redundant variables.
rare_analyser(df, cat_cols, 'SalePrice')
cat_drop_list = ['MiscFeature', 'PoolQC', 'Utilities', 'Street']

# update cat_cols list
cat_cols = [col for col in cat_cols if col not in cat_drop_list]

# drop drop list columns from dataset
for col in cat_drop_list:
    df.drop(col, axis=1, inplace=True)

# lets examine numerical variables

for col in num_cols:
    print(col, len(df[col].value_counts()))

# Below three contains only inf variables so we can remove them.
num_drop_list = ['LowQualFinSF', '3SsnPorch', 'MiscVal']

num_cols = [col for col in num_cols if col not in num_drop_list]

# drop drop list columns from dataset
for col in num_drop_list:
    df.drop(col, axis=1, inplace=True)


# Missing values

# Function to find columns with null values
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)


def fill_missing_num_with_median(dataframe, numeric_columns):
    for col in numeric_columns:
        if col == 'SalePrice':
            pass
        else:
            dataframe[col].fillna(dataframe[col].median(), inplace=True)


fill_missing_num_with_median(df, num_cols)


def fill_missing_cat_with_mode(dataframe, categorical_columns):
    for col in categorical_columns:
        dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)


fill_missing_cat_with_mode(df, cat_cols)


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

# Standardization
# It can be applied only numeric columns
cols_need_scale = [col for col in num_cols if col not in ("Id", "SalePrice")]


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


# LABEL ENCODING & ONE HOT ENCODING

# Lets define a function to create one_hot_encoding columns for categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)

# last controls
missing_values_table(df)
has_outliers(df, num_cols)

# modeling

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

# save clean train and test files to as pickle file.
train_df.to_pickle('/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/processed/train_df.pkl')
test_df.to_pickle('/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/processed/test_df.pkl')

X = train_df.drop('SalePrice', axis=1)
y = train_df[["SalePrice"]]

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
