import pandas as pd
import unittest

from sampler import MLSampler, Sampler

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='https://raw.githubusercontent.com/GianniBalistreri/happy_learning/master/test/data/avocado.csv')
print(DATA_SET['type'].value_counts(normalize=True))

