# this file is to demonstrate the dvc 

from logger import logger
import os
import sys
import pandas as pd
import numpy as np
import typing 
import path




def get_data(filepath: str) -> pd.DataFrame:
    """
    Returns the dataset as csv file
    """
    try:
        logger.debug('Reading data from {0}'.format(filepath))
        logger.debug('reading data %s'%filepath)
        data = pd.read_csv(filepath)
        logger.warning("Data loaded successfully")
        return data
    except Exception as e:
        #logger.error('error in reading data')
        logger.error("Error reading data from %s"%filepath)

if __name__ == "__main__":
    try:
        logger.info('started main program')
        filepath = 'Housing.csv'
        data = get_data(filepath)
        logger.warning('completed reading data')
    except Exception as e:
        logger.critical('Failed to complete main program')