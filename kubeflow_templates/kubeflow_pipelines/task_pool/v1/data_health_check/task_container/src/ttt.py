
import random
import string
import numpy as np
a = np.array([1,2,3,4,5,6,7,87,8,89,7,6,5,4,3,3,2])
b= np.array_split(ary=a, indices_or_sections=4)

print(b)

d = []
for i in b:
    d.append(i.tolist())

print(d)

import pandas as pd

df = pd.DataFrame(data={'A': [1,2,3,4,4,5], 'B': [2,2,2,2,2,2]})

print(type(df.index.values))

print(df.columns.values)

l = ['z', 'b', 'a', 'c']

print(sorted(l))

ss = f'{string.ascii_lowercase}{string.digits}'
print(ss)
print(''.join(random.choices(ss, k=25)))

