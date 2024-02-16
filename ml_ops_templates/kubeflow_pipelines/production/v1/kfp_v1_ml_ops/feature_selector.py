"""

Kubeflow Pipeline Component: Feature Selector

"""

from .container_op_parameters import add_container_op_parameters
from kfp import dsl
from typing import List


def feature_selector(ml_type: str,
                     train_data_set_path: str,
                     test_data_set_path: str,
                     target_feature: str,
                     aws_account_id: str,
                     aws_region: str,
                     features: List[str] = None,
                     init_pairs: int = 3,
                     init_games: int = 5,
                     increasing_pair_size_factor: float = 0.05,
                     games: int = 3,
                     penalty_factor: float = 0.1,
                     max_iter: int = 50,
                     max_players: int = -1,
                     imp_threshold: float = 0.01,
                     redundant_threshold: float = 0.01,
                     top_n_imp_features_proportion: float = 0.1,
                     feature_selection_algorithm: str = 'feature_addition',
                     feature_selection_early_stopping: bool = False,
                     model_name: str = 'xgb',
                     model_param_path: str = None,
                     aggregate_feature_imp: dict = None,
                     sep: str = ',',
                     output_path_imp_features: str = 'imp_features.json',
                     s3_output_path_metadata: str = None,
                     s3_output_path_visualization_data: str = None,
                     docker_image_name: str = 'ml-ops-feature-selector',
                     docker_image_tag: str = 'v1',
                     volume: dsl.VolumeOp = None,
                     volume_dir: str = '/mnt',
                     display_name: str = 'Feature Selector',
                     n_cpu_request: str = None,
                     n_cpu_limit: str = None,
                     n_gpu: str = None,
                     gpu_vendor: str = 'nvidia',
                     memory_request: str = '1G',
                     memory_limit: str = None,
                     ephemeral_storage_request: str = '5G',
                     ephemeral_storage_limit: str = None,
                     instance_name: str = 'm5.xlarge',
                     max_cache_staleness: str = 'P0D'
                     ) -> dsl.ContainerOp:
    """
    Generate container operator for feature selector

    :param ml_type: str
        Abbreviated name of the machine learning type:
            -> reg: Regression
            -> clf_binary: Binary Classification
            -> clf_multi: Multi-Classification

    :param train_data_set_path: str
        Complete file path of the training data set

    :param test_data_set_path: str
        Complete file path of the test data set

    :param target_feature: str
        Name of the target feature

    :param aws_account_id: str
        AWS account id

    :param aws_region: str
        AWS region name

    :param features: List[str]
        Name of the features

    :param init_pairs: int
        Number of feature pairs in initial penalty

    :param init_games: int
        Number of the games each interation in initial penalty

    :param increasing_pair_size_factor: float
        Factor of increasing number of feature pairs each iteration during tournament

    :param games: int
        Number of games each iteration of the tournament

    :param penalty_factor: float
        Factor of removing features during penalty because of poor contribution to fitness metric

    :param max_iter: int
        Maximum number of iterations during tournament

    :param max_players: int
        Maximum number of players during tournament

    :param imp_threshold: float

    :param redundant_threshold: float
        Threshold to decide whether a feature is redundant or not

    :param top_n_imp_features_proportion: float
        Proportion of top n features to select a feature used in filter-based method

    :param feature_selection_algorithm: str
        Name of the feature selection algorithm to use
            -> feature_addition: Feature addition
            -> filter-based: Filter-based
            -> recursive_feature_elimination: Recursive feature elimination (RFE)

    :param feature_selection_early_stopping: bool
        Whether to stop early if hit redundant threshold applying feature selection algorithm

    :param model_name: str
        Abbreviated name of the supervised machine learning model
            -> cat: CatBoost
            -> gbo: Gradient Boosting Decision Tree
            -> rf: Random Forest
            -> xgb: Extreme Gradient Boosting Decision Tree

    :param model_param_path: str
        Complete file path of the pre-defined model hyperparameter

    :param aggregate_feature_imp: dict
        Relationship mapping of features to aggregate feature importance score

    :param sep: str
        Separator

    :param output_path_imp_features: str
        Output path of the important features

    :param s3_output_path_metadata: str
        Complete file path of the metadata output

    :param s3_output_path_visualization_data: str
        Complete file path of the visualization output

    :param aws_account_id: str
        AWS account id

    :param docker_image_name: str
        Name of the docker image repository

    :param docker_image_tag: str
        Name of the docker image tag

    :param volume: dsl.VolumeOp
        Attached container volume

    :param volume_dir: str
        Name of the volume directory

    :param display_name: str
        Display name of the Kubeflow Pipeline component

    :param n_cpu_request: str
        Number of requested CPU's

    :param n_cpu_limit: str
        Maximum number of requested CPU's

    :param n_gpu: str
        Maximum number of requested GPU's

    :param gpu_vendor: str
        Name of the GPU vendor
            -> amd: AMD
            -> nvidia: NVIDIA

    :param memory_request: str
        Memory request

    :param memory_limit: str
        Limit of the requested memory

    :param ephemeral_storage_request: str
        Ephemeral storage request (cloud based additional memory storage)

    :param ephemeral_storage_limit: str
        Limit of the requested ephemeral storage (cloud based additional memory storage)

    :param instance_name: str
        Name of the used AWS instance (value)

    :param max_cache_staleness: str
        Maximum of staleness days of the component cache

    :return: dsl.ContainerOp
        Container operator for feature selector
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-ml_type', ml_type,
                        '-train_data_set_path', train_data_set_path,
                        '-test_data_set_path', test_data_set_path,
                        '-target_feature', target_feature,
                        '-output_path_imp_features', output_path_imp_features,
                        '-features', features,
                        '-init_pairs', init_pairs,
                        '-init_games', init_games,
                        '-increasing_pair_size_factor', increasing_pair_size_factor,
                        '-games', games,
                        '-penalty_factor', penalty_factor,
                        '-max_iter', max_iter,
                        '-max_players', max_players,
                        '-imp_threshold', imp_threshold,
                        '-redundant_threshold', redundant_threshold,
                        '-top_n_imp_features_proportion', top_n_imp_features_proportion,
                        '-feature_selection_algorithm', feature_selection_algorithm,
                        '-feature_selection_early_stopping', int(feature_selection_early_stopping),
                        '-model_name', model_name,
                        '-sep', sep
                        ]
    if model_param_path is not None:
        _arguments.extend(['-model_param_path', model_param_path])
    if aggregate_feature_imp is not None:
        _arguments.extend(['-aggregate_feature_imp', aggregate_feature_imp])
    if s3_output_path_metadata is not None:
        _arguments.extend(['-s3_output_path_metadata', s3_output_path_metadata])
    if s3_output_path_visualization_data is not None:
        _arguments.extend(['-s3_output_path_visualization_data', s3_output_path_visualization_data])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='feature_selector',
                                             image=f'{aws_account_id}.dkr.ecr.{aws_region}.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'imp_features': output_path_imp_features},
                                             output_artifact_paths=None,
                                             is_exit_handler=False,
                                             pvolumes=volume if volume is None else _volume
                                             )
    _task.set_display_name(display_name)
    add_container_op_parameters(container_op=_task,
                                n_cpu_request=n_cpu_request,
                                n_cpu_limit=n_cpu_limit,
                                n_gpu=n_gpu,
                                gpu_vendor=gpu_vendor,
                                memory_request=memory_request,
                                memory_limit=memory_limit,
                                ephemeral_storage_request=ephemeral_storage_request,
                                ephemeral_storage_limit=ephemeral_storage_limit,
                                instance_name=instance_name,
                                max_cache_staleness=max_cache_staleness
                                )
    return _task
