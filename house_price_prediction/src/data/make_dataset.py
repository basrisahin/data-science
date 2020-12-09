# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def load_fix_house_price(class_count=20):
    train = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/train.csv")
    test = pd.read_csv("/Users/basrisahin/Documents/GitHub/data-science/house_price_prediction/data/raw/test.csv")
    data = train.append(test).reset_index()

    cat_seen_as_numeric = [col for col in data.columns if data[col].dtypes != 'O'
                           and len(data[col].value_counts()) < class_count]

    for col in cat_seen_as_numeric:
        data[col] = data[col].astype(object)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
