"""

Kubeflow Pipeline: Occupancy

"""

import argparse

from kfp import dsl
from shopware_kfp_utils.analytical_data_types import analytical_data_types
from shopware_kfp_utils.data_health_check import data_health_check
from shopware_kfp_utils.data_typing import data_typing
from shopware_kfp_utils.display_visualization import display_visualization
from shopware_kfp_utils.experiment import KubeflowExperiment
from shopware_kfp_utils.interactive_visualizer import interactive_visualizer
from shopware_kfp_utils.serializer import serializer


PARSER = argparse.ArgumentParser(description="occupancy_data_analysis")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='aws account id')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='aws region name')
PARSER.add_argument('-kf_url', type=str, required=True, default=None, help='complete url of the kubeflow deployment')
PARSER.add_argument('-kf_user_name', type=str, required=True, default=None, help='kubeflow user name')
PARSER.add_argument('-kf_user_pwd', type=str, required=True, default=None, help='kubeflow user password')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default='production', help='kubeflow user namespace')
PARSER.add_argument('-kf_pipeline_name', type=str, required=False, default='occupancy_data_analysis', help='kubeflow pipeline name')
PARSER.add_argument('-kf_experiment_name', type=str, required=False, default='occupancy', help='kubeflow experiment name')
PARSER.add_argument('-kf_experiment_description', type=str, required=False, default='show case: end-to-end ml-ops pipeline', help='experiment description')
PARSER.add_argument('-kf_experiment_run_name', type=str, required=False, default='data analysis', help='name of the kubeflow experiment run')
PARSER.add_argument('-kf_enable_caching', type=int, required=False, default=0, help='whether to enable kubeflow pipeline component caching or not')
PARSER.add_argument('-recurring', type=int, required=False, default=0, help='whether to implement continuous training or not')
PARSER.add_argument('-recurring_start_time', type=str, required=False, default=None, help='start time of the recurring run')
PARSER.add_argument('-recurring_end_time', type=str, required=False, default=None, help='end time of the recurring run')
PARSER.add_argument('-recurring_interval_second', type=int, required=False, default=None, help='time interval in seconds of the recurring run')
PARSER.add_argument('-recurring_cron_expression', type=str, required=False, default='0 0 0 * * *', help='cron job expression for the recurring run')
PARSER.add_argument('-recurring_no_catchup', type=int, required=False, default=1, help='no catchup of recurring run')
PARSER.add_argument('-recurring_enable', type=int, required=False, default=1, help='whether to enable recurring run or not')
PARSER.add_argument('-recurring_job_name', type=str, required=False, default=None, help='name of the recurring run')
PARSER.add_argument('-recurring_job_description', type=str, required=False, default=None, help='description of the recurring run')
PARSER.add_argument('-auth_service_provider', type=str, required=False, default='dex', help='name of the authentication provider')
ARGS = PARSER.parse_args()


@dsl.pipeline(
    name='show_case_occupancy',
    description='End-to-end training pipeline for predicting occupancy'
)
def pipeline(aws_account_id: str, aws_region: str) -> None:
    """
    Generate continuous training Kubeflow pipeline

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        Name of the AWS region
    """
    _project_name: str = 'occupancy'
    _s3_output_bucket_name: str = 'gfb-ml-ops-interim-prod'
    _basic_path: str = f's3://{_s3_output_bucket_name}/projects/{_project_name}/'
    _processed_data_set_path: str = f'{_basic_path}processed_data_sets/'
    _data_analysis_path: str = f'{_basic_path}data_analysis/'
    _visualization_path: str = f'{_data_analysis_path}visualization/'
    _train_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatraining.txt'
    _test_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest.txt'
    _val_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest2.txt'
    _sep: str = ','
    _label_feature: str = 'train_test_split'
    _merged_data_set_path: str = f'{_processed_data_set_path}occupancy_data.json'
    _analytical_data_types_path: str = f'{_data_analysis_path}analytical_data_types.json'
    _target_feature: str = 'Occupancy'
    _ml_type: str = 'clf_binary'
    _task_0: dsl.ContainerOp = serializer(action='cases',
                                          parallelized_obj=[_train_data_set_path, _test_data_set_path, _val_data_set_path],
                                          aws_account_id=aws_account_id,
                                          aws_region=aws_region,
                                          label_feature_name=_label_feature,
                                          labels=['train', 'test', 'val'],
                                          s3_output_file_path_parallelized_data=_merged_data_set_path,
                                          docker_image_tag='v1',
                                          display_name='Merge Data Set',
                                          max_cache_staleness='P1D'
                                          )
    _task_1: dsl.ContainerOp = analytical_data_types(data_set_path=_merged_data_set_path,
                                                     s3_output_file_path_analytical_data_types=_analytical_data_types_path,
                                                     aws_account_id=aws_account_id,
                                                     aws_region=aws_region,
                                                     max_categories=100,
                                                     date_edges=None,
                                                     categorical=None,
                                                     ordinal=None,
                                                     continuous=None,
                                                     date=None,
                                                     id_text=None,
                                                     sep=_sep,
                                                     docker_image_tag='v1',
                                                     display_name='Analytical Data Types',
                                                     max_cache_staleness='P1D'
                                                     )
    _task_1.after(_task_0)
    _task_2: dsl.ContainerOp = data_health_check(data_set_path=_merged_data_set_path,
                                                 analytical_data_types_path=_analytical_data_types_path,
                                                 aws_account_id=aws_account_id,
                                                 aws_region=aws_region,
                                                 missing_value_threshold=0.95,
                                                 sep=_sep,
                                                 s3_output_file_path_data_health_check=None,
                                                 docker_image_tag='v1',
                                                 display_name='Data Health Check',
                                                 max_cache_staleness='P1D'
                                                 )
    _task_2.after(_task_1)
    with dsl.Condition(condition=_task_2.outputs['n_valid_features'] > 1, name='Less-Valid-Features'):
        _typed_data_set_path: str = f'{_processed_data_set_path}occupancy_data_typed.json'
        _task_3: dsl.ContainerOp = data_typing(data_set_path=_merged_data_set_path,
                                               analytical_data_types_path=_analytical_data_types_path,
                                               s3_output_file_path_data_set=_typed_data_set_path,
                                               missing_value_features=_task_2.outputs['missing_data'],
                                               aws_account_id=aws_account_id,
                                               aws_region=aws_region,
                                               data_types_config=None,
                                               sep=_sep,
                                               s3_output_file_path_data_typing=None,
                                               docker_image_tag='v1',
                                               display_name='Data Typing',
                                               max_cache_staleness='P1D'
                                               )
        _task_viz_0: dsl.ContainerOp = interactive_visualizer(data_set_path=_typed_data_set_path,
                                                              aws_account_id=aws_account_id,
                                                              aws_region=aws_region,
                                                              plot_type='pie',
                                                              s3_output_image_path=f'{_visualization_path}occupancy_pie.html',
                                                              title='Distribution of Occupancy (Target Feature)',
                                                              features=[_target_feature],
                                                              time_features=None,
                                                              graph_features=None,
                                                              group_by=None,
                                                              melt=False,
                                                              brushing=False,
                                                              xaxis_label=None,
                                                              yaxis_label=None,
                                                              zaxis_label=None,
                                                              annotations=None,
                                                              width=500,
                                                              height=500,
                                                              unit='px',
                                                              use_auto_extensions=False,
                                                              color_scale=None,
                                                              color_edges=None,
                                                              color_feature=None,
                                                              analytical_data_types_path=_analytical_data_types_path,
                                                              subplots_file_path=None,
                                                              sep=_sep,
                                                              docker_image_tag='v1',
                                                              display_name='Generate Pie Chart',
                                                              max_cache_staleness='P1D'
                                                              )
        _task_viz_0.after(_task_3)
        _task_viz_1: dsl.ContainerOp = interactive_visualizer(data_set_path=_typed_data_set_path,
                                                              aws_account_id=aws_account_id,
                                                              aws_region=aws_region,
                                                              plot_type='violin',
                                                              s3_output_image_path=f'{_visualization_path}occupancy_violin.html',
                                                              title='',
                                                              features=_task_1.outputs['continuous_features'],
                                                              time_features=None,
                                                              graph_features=None,
                                                              group_by=[_target_feature],
                                                              melt=True,
                                                              brushing=False,
                                                              xaxis_label=None,
                                                              yaxis_label=None,
                                                              zaxis_label=None,
                                                              annotations=None,
                                                              width=500,
                                                              height=500,
                                                              unit='px',
                                                              use_auto_extensions=False,
                                                              color_scale=None,
                                                              color_edges=None,
                                                              color_feature=None,
                                                              analytical_data_types_path=_analytical_data_types_path,
                                                              subplots_file_path=None,
                                                              sep=_sep,
                                                              docker_image_tag='v1',
                                                              display_name='Generate Violin Charts',
                                                              max_cache_staleness='P1D'
                                                              )
        _task_viz_1.after(_task_3)
        _task_4: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_viz_0.outputs['file_paths'],
                                                                         b=_task_viz_1.outputs['file_paths'],
                                                                         ),
                                                         display_name='Display Target Feature Charts',
                                                         max_cache_staleness='P1D'
                                                         )


if __name__ == '__main__':
    _kubeflow_experiment: KubeflowExperiment = KubeflowExperiment(kf_url=ARGS.kf_url,
                                                                  kf_user_name=ARGS.kf_user_name,
                                                                  kf_user_pwd=ARGS.kf_user_pwd,
                                                                  kf_user_namespace=ARGS.kf_user_namespace,
                                                                  kf_pipeline_name=ARGS.kf_pipeline_name,
                                                                  kf_experiment_name=ARGS.kf_experiment_name,
                                                                  kf_experiment_description=ARGS.kf_experiment_description,
                                                                  kf_experiment_run_name=ARGS.kf_experiment_run_name,
                                                                  kf_enable_caching=ARGS.kf_enable_caching,
                                                                  recurring=bool(ARGS.recurring),
                                                                  recurring_start_time=ARGS.recurring_start_time,
                                                                  recurring_end_time=ARGS.recurring_end_time,
                                                                  recurring_interval_second=ARGS.recurring_interval_second,
                                                                  recurring_cron_expression=ARGS.recurring_cron_expression,
                                                                  recurring_no_catchup=bool(ARGS.recurring_no_catchup),
                                                                  recurring_enable=bool(ARGS.recurring_enable),
                                                                  recurring_job_name=ARGS.recurring_job_name,
                                                                  recurring_job_description=ARGS.recurring_job_description,
                                                                  auth_service_provider=ARGS.auth_service_provider
                                                                  )
    _kubeflow_experiment.main(pipeline=pipeline, arguments=dict(aws_account_id=ARGS.aws_account_id, aws_region=ARGS.aws_region))
