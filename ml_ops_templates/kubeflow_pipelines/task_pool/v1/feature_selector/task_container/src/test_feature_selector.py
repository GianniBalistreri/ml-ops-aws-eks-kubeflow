
import pandas as pd
from supervised_machine_learning import Regression

df = pd.read_json('/Users/giannibalistreri/PycharmProjects/data_pipeline/train_data_set.json')

d = df.to_dict()

df_new = pd.DataFrame(data=d)
print(df_new)

reg = Regression()
reg.generate()
print(reg.model.__dict__.items())
for param in reg.model.__dict__.items():
    print(param)
