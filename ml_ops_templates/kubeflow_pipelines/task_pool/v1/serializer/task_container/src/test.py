
#from serializer import Serializer
import json
from typing import Dict, List
import copy


with open('/Users/giannibalistreri/PycharmProjects/data_pipeline/analytical_data_types_0.json') as file:
    a = json.load(file)
print('0', a)

with open('/Users/giannibalistreri/PycharmProjects/data_pipeline/analytical_data_types_1.json') as file:
    a = json.load(file)
print('1', a)

with open('/Users/giannibalistreri/PycharmProjects/data_pipeline/analytical_data_types_2.json') as file:
    a = json.load(file)
print('2', a)

with open('/Users/giannibalistreri/PycharmProjects/data_pipeline/analytical_data_types_3.json') as file:
    a = json.load(file)
print('3', a)


_analytical_data_types: Dict[str, List[str]] = {}
for file_path in [0, 1, 2, 3]:
    with open(f'/Users/giannibalistreri/PycharmProjects/data_pipeline/analytical_data_types_{file_path}.json') as file:
        _analytical_data_types_chunk: Dict[str, List[str]] = json.load(file)
    for analytical_data_type in _analytical_data_types_chunk.keys():
        if analytical_data_type in _analytical_data_types.keys():
            _analytical_data_types[analytical_data_type].extend(_analytical_data_types_chunk[analytical_data_type])
        else:
            _analytical_data_types.update({analytical_data_type: _analytical_data_types_chunk[analytical_data_type]})
        print(analytical_data_type)
        print(_analytical_data_types_chunk[analytical_data_type])
        _analytical_data_types[analytical_data_type] = copy.deepcopy(list(set(_analytical_data_types[analytical_data_type])))

print(_analytical_data_types)
print(_analytical_data_types.get('categorical'))
