import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_rfm_data():
    df = pd.read_csv('/Users/basrisahin/Documents/GitHub/data-science/RMF/data/online_retail_2019_2020.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['TotalPrice'] = df['Quantity'] * df['Price']
    return df


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


def num_summary(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")
