from bitarray import bitarray
from bitarray.util import ba2int
import random
import pandas as pd


class Wisard:
    def __init__(self, num_classes: int, n: int, seed:int=28) -> None:
        '''
        entry_size: Size of the binary entry
        num_classes: Number of classes on the dataset
        n: Number of RAMs
        seed: Random seed
        '''
        self.n = n 
        self.m = None # The value will be set in training
        self.num_classes = num_classes
        self.seed = seed

        self.discriminators = [Discriminator(self.n) for _ in range(self.num_classes)]

        pass

    def train(self, x: pd.DataFrame, y: pd.DataFrame) -> None:

        data_list = x.to_string(index=False, header=False).split("\n")
        class_list = y.to_string(index=False, header=False).split("\n")

        if not self.m:
            self.m = len(bitarray(data_list[0])) // self.n
            for ds in self.discriminators:
                ds.m = self.m

        for i in range(len(data_list)):
            data = data_list[i]
            random.seed(self.seed)
            random.shuffle(list(data))
            data = bitarray(data)

            self.discriminators[int(class_list[i])].train(data)

    def classify(self, x: pd.DataFrame) -> list[int]:
        data_list = x.to_string(index=False, header=False).split("\n")
        pds = []

        for data in data_list:
            random.seed(self.seed)
            random.shuffle(list(data))
            data = bitarray(data)

            data_pds = [disc.classify(data) for disc in self.discriminators]
            # print(data_pds)

            pds.append(data_pds.index(max(data_pds)))

        return pds





class Discriminator:
    def __init__(self, n) -> None:
        '''
        n: Number of RAMs
        m: Size of each tuple
        '''
        self.n = n
        self.m = None # Will be set in training

        self.RAMs = [dict() for _ in range(n)]


    def train(self, data: bitarray) -> None:
        '''
        data: Representation of the data
        '''

        for i in range(self.n):
            self.RAMs[i][ba2int(data[i*self.m: min((i+1)*self.m,len(data))])] = True

    def classify(self, data: bitarray) -> int:
        pd = 0

        for i in range(self.n):
            if self.RAMs[i].get(ba2int(data[i*self.m: (i+1)*self.m])):
                pd += 1

        return pd