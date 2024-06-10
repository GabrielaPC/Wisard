import pandas as pd

import math

class Thermometer():
    def __init__(self, data: pd.DataFrame, num_bits: int) -> None:

        self.data = data
        self.num_bits = num_bits

        pass

    def encode(self, features):
        ''' Ispired by: https://github.com/leandro-santiago/bloomwisard/blob/master/encoding/thermometer.py
        '''

        for feature in features:
            labels = self.data[feature].unique()
            min_value = min(labels)

            interval = max(labels) - min_value

            for x in labels:
                bits_activated = int(math.ceil(((x - min_value)/ interval ) * self.num_bits))
                
                self.data.loc[self.data[feature]==x, feature] = "".join(["1"]*bits_activated) + "".join(["0"]*(self.num_bits - bits_activated))