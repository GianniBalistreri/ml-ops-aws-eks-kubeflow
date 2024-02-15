
import pandas as pd
import json
import copy

a=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
a.reverse()
b=copy.deepcopy(a)

for i in a:
    print(b[b.index(i)])


