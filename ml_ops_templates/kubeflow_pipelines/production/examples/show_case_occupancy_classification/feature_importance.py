"""

Kubeflow Pipeline: Occupancy

"""

import argparse

from kfp import dsl
from shopware_kfp_utils.data_health_check import data_health_check
from shopware_kfp_utils.display_visualization import display_visualization
from shopware_kfp_utils.experiment import KubeflowExperiment
from shopware_kfp_utils.feature_engineer import feature_engineer
from shopware_kfp_utils.feature_selector import feature_selector
from shopware_kfp_utils.interactive_visualizer import interactive_visualizer
from shopware_kfp_utils.parallelizer import parallelizer


PARSER = argparse.ArgumentParser(description="occupancy_feature_importance")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='aws account id')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='aws region name')
PARSER.add_argument('-kf_url', type=str, required=True, default=None, help='complete url of the kubeflow deployment')
PARSER.add_argument('-kf_user_name', type=str, required=True, default=None, help='kubeflow user name')
PARSER.add_argument('-kf_user_pwd', type=str, required=True, default=None, help='kubeflow user password')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default=None, help='kubeflow user namespace')
PARSER.add_argument('-kf_pipeline_name', type=str, required=False, default='occupancy_feature_importance', help='kubeflow pipeline name')
PARSER.add_argument('-kf_experiment_name', type=str, required=False, default='occupancy', help='kubeflow experiment name')
PARSER.add_argument('-kf_experiment_description', type=str, required=False, default='show case: end-to-end ml-ops pipeline', help='experiment description')
PARSER.add_argument('-kf_experiment_run_name', type=str, required=False, default='feature importance', help='name of the kubeflow experiment run')
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
    _train_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatraining.txt'
    _test_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatest.txt'
    _val_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatest2.txt'
    _sep: str = ','
    _label_feature: str = 'train_test_split'
    _merged_data_set_path: str = f'{_processed_data_set_path}occupancy_data.json'
    _analytical_data_types_path: str = f'{_data_analysis_path}analytical_data_types.json'
    _target_feature: str = 'Occupancy'
    _ml_type: str = 'clf_binary'
    _processor_memory_path: str = f'{_processed_data_set_path}processor_memory.p'
    _engineered_data_set_path: str = f'{_processed_data_set_path}engineered_data.json'
    _typed_data_set_path: str = f'{_processed_data_set_path}occupancy_data_typed.json'
    _task_5: dsl.ContainerOp = feature_engineer(data_set_path=_typed_data_set_path,
                                                analytical_data_types_path=_analytical_data_types_path,
                                                target_feature=_target_feature,
                                                s3_output_file_path_data_set=_engineered_data_set_path,
                                                s3_output_file_path_processor_memory=_processor_memory_path,
                                                aws_account_id=aws_account_id,
                                                aws_region=aws_region,
                                                re_engineering=False,
                                                next_level=False,
                                                feature_engineering_config=None,
                                                features=None,
                                                ignore_features=[_label_feature],
                                                exclude_features=None,
                                                exclude_original_data=False,
                                                exclude_meth=None,
                                                use_only_meth=['one_hot_encoder',
                                                               'date_categorizer'
                                                               ],
                                                sep=_sep,
                                                parallel_mode=False,
                                                docker_image_tag='v1',
                                                max_cache_staleness='P1D'
                                                )
    _task_5_1: dsl.ContainerOp = data_health_check(data_set_path=_engineered_data_set_path,
                                                   analytical_data_types_path=_analytical_data_types_path,
                                                   aws_account_id=aws_account_id,
                                                   aws_region=aws_region,
                                                   features=_task_5.outputs['features'],
                                                   missing_value_threshold=0.95,
                                                   sep=_sep,
                                                   s3_output_file_path_data_health_check=None,
                                                   docker_image_tag='v1',
                                                   display_name='Engineered Feature Health Check',
                                                   max_cache_staleness='P1D'
                                                   )
    _task_6: dsl.ContainerOp = parallelizer(action='cases',
                                            aws_account_id=aws_account_id,
                                            aws_region=aws_region,
                                            analytical_data_types_path=_analytical_data_types_path,
                                            data_file_path=_engineered_data_set_path,
                                            split_by=_label_feature,
                                            docker_image_tag='v1',
                                            display_name='Train-Test Split',
                                            max_cache_staleness='P1D'
                                            )
    _task_6.after(_task_5_1)
    _train_data_set_path = f'{_processed_data_set_path}engineered_data_train.json'
    _test_data_set_path = f'{_processed_data_set_path}engineered_data_test.json'
    _val_data_set_path = f'{_processed_data_set_path}engineered_data_val.json'
    _feature_importance_subplot_path: str = f'{_data_analysis_path}feature_importance.json'
    _task_7: dsl.ContainerOp = feature_selector(ml_type=_ml_type,
                                                train_data_set_path=_train_data_set_path,
                                                test_data_set_path=_test_data_set_path,
                                                target_feature=_target_feature,
                                                aws_account_id=aws_account_id,
                                                aws_region=aws_region,
                                                features=_task_5_1.outputs['valid_features'],
                                                s3_output_path_visualization_data=_feature_importance_subplot_path,
                                                docker_image_tag='v1',
                                                display_name='Feature Selector',
                                                max_cache_staleness='P1D'
                                                )
    _task_7.after(_task_6)
    _task_8: dsl.ContainerOp = interactive_visualizer(s3_output_image_path=f'{_visualization_path}',
                                                      aws_account_id=aws_account_id,
                                                      aws_region=aws_region,
                                                      subplots_file_path=_feature_importance_subplot_path,
                                                      analytical_data_types_path=_analytical_data_types_path,
                                                      docker_image_tag='v1',
                                                      display_name='Generate Feature Importance Charts',
                                                      max_cache_staleness='P1D'
                                                      )
    _task_8.after(_task_7)
    _task_viz_2: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_8.outputs['file_paths']),
                                                         display_name='Display Feature Importance',
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
