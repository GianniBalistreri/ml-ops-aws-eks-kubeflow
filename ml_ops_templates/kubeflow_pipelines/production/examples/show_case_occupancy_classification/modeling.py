"""

Kubeflow Pipeline: Occupancy

"""

import argparse

from kfp import dsl
from shopware_kfp_utils.display_ml_metrics import display_metrics
from shopware_kfp_utils.display_visualization import display_visualization
from shopware_kfp_utils.experiment import KubeflowExperiment
from shopware_kfp_utils.interactive_visualizer import interactive_visualizer
from shopware_kfp_utils.model_evaluation import evaluate_machine_learning
from shopware_kfp_utils.model_generator_supervised import generate_supervised_model


PARSER = argparse.ArgumentParser(description="occupancy_modeling")
PARSER.add_argument('-aws_account_id', type=str, required=True, default=None, help='aws account id')
PARSER.add_argument('-aws_region', type=str, required=True, default=None, help='aws region name')
PARSER.add_argument('-kf_url', type=str, required=True, default=None, help='complete url of the kubeflow deployment')
PARSER.add_argument('-kf_user_name', type=str, required=True, default=None, help='kubeflow user name')
PARSER.add_argument('-kf_user_pwd', type=str, required=True, default=None, help='kubeflow user password')
PARSER.add_argument('-kf_user_namespace', type=str, required=False, default=None, help='kubeflow user namespace')
PARSER.add_argument('-kf_pipeline_name', type=str, required=False, default='occupancy_modeling', help='kubeflow pipeline name')
PARSER.add_argument('-kf_experiment_name', type=str, required=False, default='occupancy', help='kubeflow experiment name')
PARSER.add_argument('-kf_experiment_description', type=str, required=False, default='show case: end-to-end ml-ops pipeline', help='experiment description')
PARSER.add_argument('-kf_experiment_run_name', type=str, required=False, default='model training', help='name of the kubeflow experiment run')
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
    _evolutionary_algorithm_path: str = f's{_basic_path}evolutionary_results/'
    _train_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatraining.txt'
    _test_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatest.txt'
    _val_data_set_path: str = 's3://gfb-feature-store-prod/external/occupancy/datatest2.txt'
    _sep: str = ','
    _label_feature: str = 'train_test_split'
    _merged_data_set_path: str = f'{_processed_data_set_path}occupancy_data.json'
    _analytical_data_types_path: str = f'{_data_analysis_path}analytical_data_types.json'
    _target_feature: str = 'Occupancy'
    _ml_type: str = 'clf_binary'
    _train_data_set_path = f'{_processed_data_set_path}engineered_data_train.json'
    _test_data_set_path = f'{_processed_data_set_path}engineered_data_test.json'
    _val_data_set_path = f'{_processed_data_set_path}engineered_data_val.json'
    _model_file_path: str = f'{_modeling_path}occupancy_xgb_model.joblib'
    _model_param: str = f'{_modeling_path}xgb_model_param.json'
    _metadata_file_path: str = f'{_modeling_path}xgb_model_metadata.json'
    _evaluation_train_data_file_path: str = f'{_modeling_path}xgb_model_evaluation_train.json'
    _evaluation_test_data_file_path: str = f'{_modeling_path}xgb_model_evaluation_test.json'
    _evaluation_val_data_file_path: str = f'{_modeling_path}xgb_model_evaluation_val.json'
    _evaluation_data_file_path: str = f'{_modeling_path}xgb_model_evaluation.json'
    _prediction_var_name: str = 'prediction'
    _task_9: dsl.ContainerOp = generate_supervised_model(ml_type=_ml_type,
                                                         model_name='xgb',
                                                         target_feature=_target_feature,
                                                         train_data_set_path=_train_data_set_path,
                                                         test_data_set_path=_test_data_set_path,
                                                         val_data_set_path=_val_data_set_path,
                                                         predictors=['Light', 'CO2'],
                                                         prediction_variable_name=_prediction_var_name,
                                                         s3_output_path_metadata=_metadata_file_path,
                                                         s3_output_path_evaluation_train_data=_evaluation_train_data_file_path,
                                                         s3_output_path_evaluation_test_data=_evaluation_test_data_file_path,
                                                         s3_output_path_evaluation_val_data=_evaluation_val_data_file_path,
                                                         s3_output_path_model=_model_file_path,
                                                         s3_output_path_param=_model_param,
                                                         aws_account_id=aws_account_id,
                                                         aws_region=aws_region,
                                                         docker_image_tag='v1',
                                                         display_name='Train XGB Classifier',
                                                         max_cache_staleness='P1D'
                                                         )
    _metrics_file_path: str = f'{_modeling_path}metrics.json'
    _metrics_visualization_file_path: str = f'{_modeling_path}metrics_visualization.json'
    _metrics_confusion_matrix_file_path: str = f'{_modeling_path}confusion_matrix.csv'
    _metrics_roc_file_path: str = f'{_modeling_path}roc_auc_curve.csv'
    _metrics_table_file_path: str = f'{_modeling_path}table_metrics.json'
    _image_output_path: str = f'{_modeling_path}metrics.html'
    _task_10: dsl.ContainerOp = evaluate_machine_learning(ml_type=_ml_type,
                                                          target_feature_name=_target_feature,
                                                          prediction_feature_name=_prediction_var_name,
                                                          train_data_set_path=_evaluation_train_data_file_path,
                                                          test_data_set_path=_evaluation_test_data_file_path,
                                                          s3_output_path_metrics=_metrics_file_path,
                                                          s3_output_path_visualization=_metrics_visualization_file_path,
                                                          s3_output_path_confusion_matrix=_metrics_confusion_matrix_file_path,
                                                          s3_output_path_roc_curve=_metrics_roc_file_path,
                                                          s3_output_path_metric_table=_metrics_table_file_path,
                                                          aws_account_id=aws_account_id,
                                                          aws_region=aws_region,
                                                          docker_image_tag='v1',
                                                          display_name='Model Evaluation',
                                                          max_cache_staleness='P1D'
                                                          )
    _task_10.after(_task_9)
    _task_viz_3: dsl.ContainerOp = interactive_visualizer(s3_output_image_path=_image_output_path,
                                                          subplots_file_path=_metrics_visualization_file_path,
                                                          aws_account_id=aws_account_id,
                                                          aws_region=aws_region,
                                                          docker_image_tag='v1',
                                                          max_cache_staleness='P1D'
                                                          )
    _task_viz_3.after(_task_10)
    _task_viz_4: dsl.ContainerOp = display_visualization(file_paths=dict(a=_task_viz_3.outputs['file_paths']),
                                                         display_name='Display Classification Charts',
                                                         max_cache_staleness='P1D'
                                                         )
    _task_viz_5: dsl.ContainerOp = display_metrics(file_paths=[_metrics_confusion_matrix_file_path,
                                                               _metrics_table_file_path
                                                               ],
                                                   metric_types=['confusion_matrix', 'table'],
                                                   header=['index', 'metric_name', 'metric_value', 'data_set'],
                                                   target_feature=_target_feature,
                                                   prediction_feature=_prediction_var_name,
                                                   labels=['0', '1'],
                                                   display_name='Display Classification Metrics',
                                                   max_cache_staleness='P1D'
                                                   )
    _task_viz_5.after(_task_10)
    _image_explain_results: str = f'{_visualization_path}explain_prediction.html'
    _task_viz_6: dsl.ContainerOp = interactive_visualizer(data_set_path=_evaluation_test_data_file_path,
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
                                                          max_cache_staleness='P1D'
                                                          )
    _task_viz_6.after(_task_10)
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
