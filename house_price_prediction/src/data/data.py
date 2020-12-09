import numpy as np
import pandas as pd

def load_fix_house_price(class_count=20):
    train = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/train.csv")
    test = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/test.csv")
    data = train.append(test).reset_index()

    cat_seen_as_numeric = [col for col in data.columns if data[col].dtypes != 'O'
                    and len(data[col].value_counts()) < class_count]

    for col in cat_seen_as_numeric:
        data[col] = data[col].astype(object)

    return data