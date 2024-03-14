"""

Kubeflow Pipeline: Occupancy

"""

import argparse

from kfp import dsl
from kfp_v1_ml_ops.analytical_data_types import analytical_data_types
from kfp_v1_ml_ops.data_health_check import data_health_check
from kfp_v1_ml_ops.data_typing import data_typing
from kfp_v1_ml_ops.display_visualization import display_visualization
from kfp_v1_ml_ops.evolutionary_algorithm import EvolutionaryAlgorithm
from kfp_v1_ml_ops.experiment import KubeflowExperiment
from kfp_v1_ml_ops.feature_engineer import feature_engineer
from kfp_v1_ml_ops.feature_selector import feature_selector
from kfp_v1_ml_ops.interactive_visualizer import interactive_visualizer
from kfp_v1_ml_ops.parallelizer import parallelizer
from kfp_v1_ml_ops.serializer import serializer


PARSER = argparse.ArgumentParser(description="occupancy_training")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='aws account id')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='aws region name')
PARSER.add_argument('-kf_url', type=str, required=True, default=None, help='complete url of the kubeflow deployment')
PARSER.add_argument('-kf_user_name', type=str, required=True, default=None, help='kubeflow user name')
PARSER.add_argument('-kf_user_pwd', type=str, required=True, default=None, help='kubeflow user password')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default=None, help='kubeflow user namespace')
PARSER.add_argument('-kf_pipeline_name', type=str, required=False, default=None, help='kubeflow pipeline name')
PARSER.add_argument('-kf_experiment_name', type=str, required=False, default='occupancy', help='kubeflow experiment name')
PARSER.add_argument('-kf_experiment_description', type=str, required=False, default='show case: end-to-end ml-ops pipeline', help='experiment description')
PARSER.add_argument('-kf_experiment_run_name', type=str, required=False, default='continuous training', help='name of the kubeflow experiment run')
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
    _modeling_path: str = f'{_basic_path}modeling/'
    _evolutionary_algorithm_path: str = f'{_basic_path}evolutionary_results/'
    _train_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatraining.txt'
    _test_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest.txt'
    _val_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest2.txt'
    _sep: str = ','
    _label_feature: str = 'train_test_split'
    _merged_data_set_path: str = f'{_processed_data_set_path}occupancy_data.json'
    _analytical_data_types_path: str = f'{_data_analysis_path}analytical_data_types.json'
    _target_feature: str = 'Occupancy'
    _ml_type: str = 'clf_binary'
    _metadata_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_metadata.json'
    _model_file_path: str = f'{_evolutionary_algorithm_path}'
    _model_param: str = f'{_evolutionary_algorithm_path}'
    _metadata_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_metadata.json'
    _evaluation_train_data_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_evaluation_train.json'
    _evaluation_test_data_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_evaluation_test.json'
    _evaluation_val_data_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_evaluation_val.json'
    _evaluation_data_file_path: str = f'{_evolutionary_algorithm_path}xgb_model_evaluation.json'
    _prediction_var_name: str = 'prediction'
    _metrics_file_path: str = f'{_evolutionary_algorithm_path}metrics.json'
    _metrics_visualization_file_path: str = f'{_evolutionary_algorithm_path}metrics_visualization.json'
    _metrics_confusion_matrix_file_path: str = f'{_evolutionary_algorithm_path}confusion_matrix.csv'
    _metrics_table_file_path: str = f'{_evolutionary_algorithm_path}table_metrics.json'
    _image_output_path: str = f'{_evolutionary_algorithm_path}metrics.html'
    _evolutionary_algorithm_visualization_file_path: str = f'{_evolutionary_algorithm_path}evolutionary_algorithm_visualization.json'
    _evolutionary_algorithm_instructions_file_path: str = f'{_evolutionary_algorithm_path}evolutionary_algorithm_instructions.json'
    _env_reaction_file_path: str = f'{_evolutionary_algorithm_path}env_reaction.json'
    _ea_image_path: str = _evolutionary_algorithm_path
    _task_0: dsl.ContainerOp = serializer(action='cases',
                                          aws_account_id=aws_account_id,
                                          aws_region=aws_region,
                                          parallelized_obj=[_train_data_set_path, _test_data_set_path, _val_data_set_path],
                                          label_feature_name=_label_feature,
                                          labels=['train', 'test', 'val'],
                                          s3_output_file_path_parallelized_data=_merged_data_set_path,
                                          docker_image_tag='v1',
                                          display_name='Merge Data Set',
                                          max_cache_staleness='P0D'
                                          )
    _task_1: dsl.ContainerOp = analytical_data_types(data_set_path=_merged_data_set_path,
                                                     s3_output_file_path_analytical_data_types=_analytical_data_types_path,
                                                     max_categories=100,
                                                     date_edges=None,
                                                     categorical=None,
                                                     ordinal=None,
                                                     continuous=None,
                                                     date=None,
                                                     id_text=None,
                                                     sep=_sep,
                                                     aws_account_id=aws_account_id,
                                                     aws_region=aws_region,
                                                     docker_image_tag='v1',
                                                     display_name='Analytical Data Types',
                                                     max_cache_staleness='P0D'
                                                     )
    _task_1.after(_task_0)
    _task_2: dsl.ContainerOp = data_health_check(data_set_path=_merged_data_set_path,
                                                 analytical_data_types_path=_analytical_data_types_path,
                                                 missing_value_threshold=0.95,
                                                 sep=_sep,
                                                 s3_output_file_path_data_health_check=None,
                                                 aws_account_id=aws_account_id,
                                                 aws_region=aws_region,
                                                 docker_image_tag='v1',
                                                 display_name='Data Health Check',
                                                 max_cache_staleness='P0D'
                                                 )
    _task_2.after(_task_1)
    if dsl.Condition(condition=_task_2.outputs['n_valid_features'] > 1, name='Less Valid Features'):
        _typed_data_set_path: str = f'{_processed_data_set_path}occupancy_data_typed.json'
        _task_3: dsl.ContainerOp = data_typing(data_set_path=_merged_data_set_path,
                                               analytical_data_types_path=_analytical_data_types_path,
                                               s3_output_file_path_data_set=_typed_data_set_path,
                                               missing_value_features=_task_2.outputs['missing_data'],
                                               data_types_config=None,
                                               sep=_sep,
                                               s3_output_file_path_data_typing=None,
                                               aws_account_id=aws_account_id,
                                               aws_region=aws_region,
                                               docker_image_tag='v1',
                                               display_name='Data Typing',
                                               max_cache_staleness='P0D'
                                               )
        _task_viz_0: dsl.ContainerOp = interactive_visualizer(data_set_path=_typed_data_set_path,
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
                                                              aws_account_id=aws_account_id,
                                                              aws_region=aws_region,
                                                              docker_image_tag='v1',
                                                              display_name='Generate Pie Chart',
                                                              max_cache_staleness='P0D'
                                                              )
        _task_viz_0.after(_task_3)
        _task_viz_1: dsl.ContainerOp = interactive_visualizer(data_set_path=_typed_data_set_path,
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
                                                              aws_account_id=aws_account_id,
                                                              aws_region=aws_region,
                                                              docker_image_tag='v1',
                                                              display_name='Generate Violin Charts',
                                                              max_cache_staleness='P0D'
                                                              )
        _task_viz_1.after(_task_3)
        _task_4: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_viz_0.outputs['file_paths'],
                                                                         b=_task_viz_1.outputs['file_paths'],
                                                                         ),
                                                         display_name='Display Target Feature Charts',
                                                         max_cache_staleness='P0D'
                                                         )
        _processor_memory_path: str = f'{_processed_data_set_path}processor_memory.p'
        _engineered_data_set_path: str = f'{_processed_data_set_path}engineered_data.json'
        _task_5: dsl.ContainerOp = feature_engineer(data_set_path=_typed_data_set_path,
                                                    analytical_data_types_path=_analytical_data_types_path,
                                                    target_feature=_target_feature,
                                                    s3_output_file_path_data_set=_engineered_data_set_path,
                                                    s3_output_file_path_processor_memory=_processor_memory_path,
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
                                                    aws_account_id=aws_account_id,
                                                    aws_region=aws_region,
                                                    docker_image_tag='v1',
                                                    max_cache_staleness='P0D'
                                                    )
        _task_5.after(_task_3)
        _task_5_1: dsl.ContainerOp = data_health_check(data_set_path=_engineered_data_set_path,
                                                       analytical_data_types_path=_analytical_data_types_path,
                                                       features=_task_5.outputs['features'],
                                                       missing_value_threshold=0.95,
                                                       sep=_sep,
                                                       s3_output_file_path_data_health_check=None,
                                                       aws_account_id=aws_account_id,
                                                       aws_region=aws_region,
                                                       docker_image_tag='v1',
                                                       display_name='Engineered Feature Health Check',
                                                       max_cache_staleness='P0D'
                                                       )
        _task_6: dsl.ContainerOp = parallelizer(action='cases',
                                                analytical_data_types_path=_analytical_data_types_path,
                                                data_file_path=_engineered_data_set_path,
                                                split_by=_label_feature,
                                                aws_account_id=aws_account_id,
                                                aws_region=aws_region,
                                                docker_image_tag='v1',
                                                display_name='Train-Test Split',
                                                max_cache_staleness='P0D'
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
                                                    features=_task_5_1.outputs['valid_features'],
                                                    s3_output_path_visualization_data=_feature_importance_subplot_path,
                                                    aws_account_id=aws_account_id,
                                                    aws_region=aws_region,
                                                    docker_image_tag='v1',
                                                    display_name='Feature Selector',
                                                    max_cache_staleness='P0D'
                                                    )
        _task_7.after(_task_6)
        _task_8: dsl.ContainerOp = interactive_visualizer(s3_output_image_path=f'{_visualization_path}',
                                                          subplots_file_path=_feature_importance_subplot_path,
                                                          analytical_data_types_path=_analytical_data_types_path,
                                                          aws_account_id=aws_account_id,
                                                          aws_region=aws_region,
                                                          docker_image_tag='v1',
                                                          display_name='Generate Feature Importance Charts',
                                                          max_cache_staleness='P0D'
                                                          )
        _task_8.after(_task_7)
        _task_viz_2: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_8.outputs['file_paths']),
                                                             display_name='Display Feature Importance',
                                                             max_cache_staleness='P0D'
                                                             )
        _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(s3_metadata_file_path=_metadata_file_path,
                                                                               ml_type=_ml_type,
                                                                               target=_target_feature,
                                                                               prediction_feature=_prediction_var_name,
                                                                               features=_task_7.outputs['imp_features'],
                                                                               models=['xgb'],
                                                                               metrics=['roc_auc'],
                                                                               metric_types=['table', 'confusion_matrix'],
                                                                               labels=['0', '1'],
                                                                               train_data_file_path=_train_data_set_path,
                                                                               test_data_file_path=_test_data_set_path,
                                                                               s3_output_file_path_generator_instructions=_evolutionary_algorithm_instructions_file_path,
                                                                               s3_output_file_path_modeling=_model_file_path,
                                                                               s3_output_file_path_evolutionary_algorithm_visualization=_evolutionary_algorithm_visualization_file_path,
                                                                               s3_output_file_path_evolutionary_algorithm_images=_ea_image_path,
                                                                               s3_output_file_path_best_model_visualization=_metrics_visualization_file_path,
                                                                               s3_output_file_path_best_model_images=_ea_image_path,
                                                                               s3_output_path_metric_table=_metrics_table_file_path,
                                                                               s3_output_path_confusion_matrix=_metrics_confusion_matrix_file_path,
                                                                               environment_reaction_path=_env_reaction_file_path,
                                                                               max_iterations=3,
                                                                               pop_size=6,
                                                                               aws_account_id=aws_account_id,
                                                                               aws_region=aws_region,
                                                                               evolutionary_algorithm_docker_image_tag='v1',
                                                                               generate_supervised_model_docker_image_tag='v1',
                                                                               evaluate_machine_learning_docker_image_tag='v1',
                                                                               serializer_docker_image_tag='v1',
                                                                               interactive_visualizer_docker_image_tag='v1'
                                                                               )
        _task_9, _task_10 = _evolutionary_algorithm.hyperparameter_tuning()
        _task_11, _task_12 = _evolutionary_algorithm.hyperparameter_tuning()
        _task_11.after(_task_10)
        _task_12.after(_task_10)
        _task_13, _task_14 = _evolutionary_algorithm.hyperparameter_tuning()
        _task_13.after(_task_12)
        _task_14.after(_task_12)
        _task_15, _task_16 = _evolutionary_algorithm.hyperparameter_tuning()
        _task_15.after(_task_14)
        _task_16.after(_task_14)
        _image_explain_results: str = f'{_visualization_path}explain_prediction.html'
        _task_viz_6: dsl.ContainerOp = interactive_visualizer(data_set_path=_task_16.outputs['evaluation_test_data_file_path'],
                                                              plot_type='parcoords',
                                                              s3_output_image_path=_image_explain_results,
                                                              title='Explain Prediction',
                                                              features=['Light',
                                                                        'CO2',
                                                                        _target_feature,
                                                                        _prediction_var_name
                                                                        ],
                                                              time_features=None,
                                                              graph_features=None,
                                                              group_by=None,
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
                                                              color_feature=_prediction_var_name,
                                                              analytical_data_types_path=_analytical_data_types_path,
                                                              subplots_file_path=None,
                                                              sep=_sep,
                                                              aws_account_id=aws_account_id,
                                                              aws_region=aws_region,
                                                              docker_image_tag='v1',
                                                              display_name='Generate Parcoords Charts',
                                                              max_cache_staleness='P0D'
                                                              )
        _task_viz_7: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_viz_6.outputs['file_paths']),
                                                             display_name='Display Explainability Chart',
                                                             max_cache_staleness='P0D'
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
