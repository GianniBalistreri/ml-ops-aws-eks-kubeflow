"""

Task: ... (Function to run in container)

"""

import boto3
import json
import pandas as pd

from batch_prediction import AnalyticalDataTypes
from typing import NamedTuple, List


def analytical_data_types(data_set_path: str,
                          feature_names: List[str],
                          output_bucket_name: str,
                          output_file_path_analytical_data_type: str,
                          sep: str = ',',
                          ) -> NamedTuple(typename='outputs', fields=[('analytical_data_type', dict)]):
    """
    Evaluate analytical data types

    :param data_set_path: str
        Complete file path of the data set

    :param feature_names: List[str]
        Name of the features

    :param output_bucket_name: str
        Name of the output S3 bucket

    :param output_file_path_analytical_data_type: str
        Path of the analytical data type information to save

    :param sep: str
        Separator

    :return: NamedTuple
        Analytical data types of given features
    """
    _df: pd.DataFrame = pd.read_csv(filepath_or_buffer=data_set_path, sep=sep)
    _analytical_data_types: AnalyticalDataTypes = AnalyticalDataTypes(df=_df,
                                                                      feature_names=feature_names,
                                                                      date_edges=None,
                                                                      max_categories=100
                                                                      )
    _analytical_data_type: dict = _analytical_data_types.main()
    _s3_resource: boto3 = boto3.resource('s3')
    _s3_model_obj: _s3_resource.Object = _s3_resource.Object(output_bucket_name, output_file_path_analytical_data_type)
    _s3_model_obj.put(Body=json.dumps(obj=_analytical_data_type))
    return [_analytical_data_type]
