"""

Kubeflow Pipeline: Occupancy

"""

import argparse

from kfp import dsl
from kfp_v1_ml_ops.display_visualization import display_visualization
from kfp_v1_ml_ops.evolutionary_algorithm import EvolutionaryAlgorithm
from kfp_v1_ml_ops.experiment import KubeflowExperiment
from kfp_v1_ml_ops.interactive_visualizer import interactive_visualizer


PARSER = argparse.ArgumentParser(description="occupancy_hyperparameter_tuning")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='aws account id')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='aws region name')
PARSER.add_argument('-kf_url', type=str, required=True, default=None, help='complete url of the kubeflow deployment')
PARSER.add_argument('-kf_user_name', type=str, required=True, default=None, help='kubeflow user name')
PARSER.add_argument('-kf_user_pwd', type=str, required=True, default=None, help='kubeflow user password')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default=None, help='kubeflow user namespace')
PARSER.add_argument('-kf_pipeline_name', type=str, required=False, default='occupancy_hyperparameter_tuning', help='kubeflow pipeline name')
PARSER.add_argument('-kf_experiment_name', type=str, required=False, default='occupancy', help='kubeflow experiment name')
PARSER.add_argument('-kf_experiment_description', type=str, required=False, default='show case: end-to-end ml-ops pipeline', help='experiment description')
PARSER.add_argument('-kf_experiment_run_name', type=str, required=False, default='hyperparameter tuning', help='name of the kubeflow experiment run')
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
    _evolutionary_algorithm_path: str = f'{_basic_path}evolutionary_results/'
    _data_analysis_path: str = f'{_basic_path}data_analysis/'
    _visualization_path: str = f'{_data_analysis_path}visualization/'
    _modeling_path: str = f'{_basic_path}modeling/'
    _train_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatraining.txt'
    _test_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest.txt'
    _val_data_set_path: str = 's3://gfb-feature-store-prod/occupancy/datatest2.txt'
    _sep: str = ','
    _target_feature: str = 'Occupancy'
    _ml_type: str = 'clf_binary'
    _analytical_data_types_path: str = f'{_data_analysis_path}analytical_data_types.json'
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
    _evolutionary_algorithm: EvolutionaryAlgorithm = EvolutionaryAlgorithm(s3_metadata_file_path=_metadata_file_path,
                                                                           ml_type=_ml_type,
                                                                           target=_target_feature,
                                                                           prediction_feature=_prediction_var_name,
                                                                           features=['Light', 'CO2'],
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
    _task_8, _task_9 = _evolutionary_algorithm.hyperparameter_tuning()
    _task_10, _task_11 = _evolutionary_algorithm.hyperparameter_tuning()
    _task_10.after(_task_9)
    _task_11.after(_task_9)
    _task_12, _task_13 = _evolutionary_algorithm.hyperparameter_tuning()
    _task_12.after(_task_11)
    _task_13.after(_task_11)
    _task_14, _task_15 = _evolutionary_algorithm.hyperparameter_tuning()
    _task_14.after(_task_13)
    _task_15.after(_task_13)
    _image_explain_results: str = f'{_visualization_path}explain_prediction.html'
    _task_viz_6: dsl.ContainerOp = interactive_visualizer(data_set_path=_task_15.outputs['evaluation_test_data_file_path'],
                                                          aws_account_id=aws_account_id,
                                                          aws_region=aws_region,
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
                                                          docker_image_tag='v1',
                                                          display_name='Generate Parcoords Charts',
                                                          max_cache_staleness='P1D'
                                                          )
    _task_viz_7: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_viz_6.outputs['file_paths']),
                                                         display_name='Display Explainability Chart',
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
