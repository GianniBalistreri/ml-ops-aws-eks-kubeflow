
from task import analytical_data_types
from kfp.components import func_to_container_op

a = func_to_container_op(func=analytical_data_types,
                         output_component_file='analytical_data_types.yaml',
                         base_image='python:3.9-alpine',
                         packages_to_install=['boto3',
                                              'pandas==2.1.0',
                                              's3fs==2023.9.2'
                                              ]
                         )
