"""

Kubeflow Pipeline Component: Evolutionary Algorithm

"""

from .container_op_parameters import add_container_op_parameters
from .display_ml_metrics import display_metrics
from .display_visualization import display_visualization
from .interactive_visualizer import interactive_visualizer
from .model_evaluation import evaluate_machine_learning
from .model_generator_supervised import generate_supervised_model
from .serializer import serializer
from kfp import dsl
from kfp.components import create_component_from_func
from typing import List, NamedTuple, Tuple, Union


class EvolutionaryAlgorithm:
    """
    Class for hyperparameter tuning Kubeflow component using evolutionary algorithm
    """
    def __init__(self,
                 s3_metadata_file_path: str,
                 ml_type: str,
                 target: str,
                 features: Union[List[str], dsl.PipelineParam],
                 models: List[str],
                 metrics: List[str],
                 train_data_file_path: str,
                 test_data_file_path: str,
                 s3_output_file_path_generator_instructions: str,
                 s3_output_file_path_modeling: str,
                 s3_output_file_path_evolutionary_algorithm_visualization: str,
                 s3_output_file_path_evolutionary_algorithm_images: str,
                 s3_output_file_path_best_model_visualization: str,
                 s3_output_file_path_best_model_images: str,
                 val_data_file_path: str = None,
                 algorithm: str = 'ga',
                 max_iterations: int = 10,
                 pop_size: int = 64,
                 burn_in_iterations: int = -1,
                 warm_start: bool = True,
                 change_rate: float = 0.1,
                 change_prob: float = 0.85,
                 parents_ratio: float = 0.5,
                 crossover: bool = True,
                 early_stopping: int = 0,
                 convergence: bool = False,
                 convergence_measure: str = 'min',
                 timer_in_seconds: int = 43200,
                 re_populate: bool = False,
                 re_populate_threshold: float = 3.0,
                 max_trials: int = 2,
                 environment_reaction_path: str = None,
                 results_table: bool = True,
                 model_distribution: bool = True,
                 model_evolution: bool = True,
                 param_distribution: bool = False,
                 train_time_distribution: bool = True,
                 breeding_map: bool = False,
                 breeding_graph: bool = False,
                 fitness_distribution: bool = True,
                 fitness_evolution: bool = True,
                 fitness_dimensions: bool = True,
                 per_iteration: bool = True,
                 aws_account_id: str = '711117404296',
                 evolutionary_algorithm_docker_image_name: str = 'ml-ops-evolutionary-algorithm',
                 evolutionary_algorithm_docker_image_tag: str = 'v1',
                 evolutionary_algorithm_volume: dsl.VolumeOp = None,
                 evolutionary_algorithm_volume_dir: str = '/mnt',
                 evolutionary_algorithm_display_name: str = 'Evolutionary Algorithm',
                 evolutionary_algorithm_n_cpu_request: str = None,
                 evolutionary_algorithm_n_cpu_limit: str = None,
                 evolutionary_algorithm_n_gpu: str = None,
                 evolutionary_algorithm_gpu_vendor: str = 'nvidia',
                 evolutionary_algorithm_memory_request: str = '100Mi',
                 evolutionary_algorithm_memory_limit: str = None,
                 evolutionary_algorithm_ephemeral_storage_request: str = '100Mi',
                 evolutionary_algorithm_ephemeral_storage_limit: str = None,
                 evolutionary_algorithm_instance_name: str = 'm5.xlarge',
                 evolutionary_algorithm_max_cache_staleness: str = 'P0D',
                 extract_instruction_python_version: str = '3.9',
                 extract_instruction_display_name: str = 'Extract Instructions',
                 extract_instruction_n_cpu_request: str = None,
                 extract_instruction_n_cpu_limit: str = None,
                 extract_instruction_n_gpu: str = None,
                 extract_instruction_gpu_vendor: str = 'nvidia',
                 extract_instruction_memory_request: str = '100Mi',
                 extract_instruction_memory_limit: str = None,
                 extract_instruction_ephemeral_storage_request: str = '100Mi',
                 extract_instruction_ephemeral_storage_limit: str = None,
                 extract_instruction_instance_name: str = 'm5.xlarge',
                 extract_instruction_max_cache_staleness: str = 'P0D',
                 generate_supervised_model_docker_image_name: str = 'ml-ops-model-generator-supervised',
                 generate_supervised_model_docker_image_tag: str = 'v1',
                 generate_supervised_model_volume: dsl.VolumeOp = None,
                 generate_supervised_model_volume_dir: str = '/mnt',
                 generate_supervised_model_display_name: str = 'Supervised Model Generator',
                 generate_supervised_model_n_cpu_request: str = None,
                 generate_supervised_model_n_cpu_limit: str = None,
                 generate_supervised_model_n_gpu: str = None,
                 generate_supervised_model_gpu_vendor: str = 'nvidia',
                 generate_supervised_model_memory_request: str = '1G',
                 generate_supervised_model_memory_limit: str = None,
                 generate_supervised_model_ephemeral_storage_request: str = '1G',
                 generate_supervised_model_ephemeral_storage_limit: str = None,
                 generate_supervised_model_instance_name: str = 'm5.xlarge',
                 generate_supervised_model_max_cache_staleness: str = 'P0D',
                 evaluate_machine_learning_docker_image_name: str = 'ml-ops-model-evaluation',
                 evaluate_machine_learning_docker_image_tag: str = 'v1',
                 evaluate_machine_learning_volume: dsl.VolumeOp = None,
                 evaluate_machine_learning_volume_dir: str = '/mnt',
                 evaluate_machine_learning_display_name: str = 'Model Evaluation',
                 evaluate_machine_learning_n_cpu_request: str = None,
                 evaluate_machine_learning_n_cpu_limit: str = None,
                 evaluate_machine_learning_n_gpu: str = None,
                 evaluate_machine_learning_gpu_vendor: str = 'nvidia',
                 evaluate_machine_learning_memory_request: str = '1G',
                 evaluate_machine_learning_memory_limit: str = None,
                 evaluate_machine_learning_ephemeral_storage_request: str = '1G',
                 evaluate_machine_learning_ephemeral_storage_limit: str = None,
                 evaluate_machine_learning_instance_name: str = 'm5.xlarge',
                 evaluate_machine_learning_max_cache_staleness: str = 'P0D',
                 serializer_docker_image_name: str = 'ml-ops-serializer',
                 serializer_docker_image_tag: str = 'v1',
                 serializer_volume: dsl.VolumeOp = None,
                 serializer_volume_dir: str = '/mnt',
                 serializer_display_name: str = 'Serializer',
                 serializer_n_cpu_request: str = None,
                 serializer_n_cpu_limit: str = None,
                 serializer_n_gpu: str = None,
                 serializer_gpu_vendor: str = 'nvidia',
                 serializer_memory_request: str = '100Mi',
                 serializer_memory_limit: str = None,
                 serializer_ephemeral_storage_request: str = '100Mi',
                 serializer_ephemeral_storage_limit: str = None,
                 serializer_instance_name: str = 'm5.xlarge',
                 serializer_max_cache_staleness: str = 'P0D',
                 gather_metadata_python_version: str = '3.9',
                 gather_metadata_display_name: str = 'Gather Metadata',
                 gather_metadata_n_cpu_request: str = None,
                 gather_metadata_n_cpu_limit: str = None,
                 gather_metadata_n_gpu: str = None,
                 gather_metadata_gpu_vendor: str = 'nvidia',
                 gather_metadata_memory_request: str = '100Mi',
                 gather_metadata_memory_limit: str = None,
                 gather_metadata_ephemeral_storage_request: str = '100Mi',
                 gather_metadata_ephemeral_storage_limit: str = None,
                 gather_metadata_instance_name: str = 'm5.xlarge',
                 gather_metadata_max_cache_staleness: str = 'P0D',
                 interactive_visualizer_docker_image_name: str = 'ml-ops-interactive-visualizer',
                 interactive_visualizer_docker_image_tag: str = 'v1',
                 interactive_visualizer_volume: dsl.VolumeOp = None,
                 interactive_visualizer_volume_dir: str = '/mnt',
                 interactive_visualizer_display_name: str = 'Interactive Visualizer',
                 interactive_visualizer_n_cpu_request: str = None,
                 interactive_visualizer_n_cpu_limit: str = None,
                 interactive_visualizer_n_gpu: str = None,
                 interactive_visualizer_gpu_vendor: str = 'nvidia',
                 interactive_visualizer_memory_request: str = '100Mi',
                 interactive_visualizer_memory_limit: str = None,
                 interactive_visualizer_ephemeral_storage_request: str = '100Mi',
                 interactive_visualizer_ephemeral_storage_limit: str = None,
                 interactive_visualizer_instance_name: str = 'm5.xlarge',
                 interactive_visualizer_max_cache_staleness: str = 'P0D',
                 display_visualization_python_version: str = '3.9',
                 display_visualization_display_name: str = 'Gather Metadata',
                 display_visualization_n_cpu_request: str = None,
                 display_visualization_n_cpu_limit: str = None,
                 display_visualization_n_gpu: str = None,
                 display_visualization_gpu_vendor: str = 'nvidia',
                 display_visualization_memory_request: str = '100Mi',
                 display_visualization_memory_limit: str = None,
                 display_visualization_ephemeral_storage_request: str = '100Mi',
                 display_visualization_ephemeral_storage_limit: str = None,
                 display_visualization_instance_name: str = 'm5.xlarge',
                 display_visualization_max_cache_staleness: str = 'P0D',
                 display_metric_python_version: str = '3.9',
                 display_metric_display_name: str = 'Gather Metadata',
                 display_metric_n_cpu_request: str = None,
                 display_metric_n_cpu_limit: str = None,
                 display_metric_n_gpu: str = None,
                 display_metric_gpu_vendor: str = 'nvidia',
                 display_metric_memory_request: str = '100Mi',
                 display_metric_memory_limit: str = None,
                 display_metric_ephemeral_storage_request: str = '100Mi',
                 display_metric_ephemeral_storage_limit: str = None,
                 display_metric_instance_name: str = 'm5.xlarge',
                 display_metric_max_cache_staleness: str = 'P0D'
                 ):
        """
        :param s3_metadata_file_path: str
            Complete file path of the metadata

        :param target: str
            Name of the target feature

        :param features: List[str]
            Name of the features

        :param models: List[str]
            Abbreviated name of the machine learning models

        :param train_data_file_path: str
            Complete file path of the training data

        :param test_data_file_path: str
            Complete file path of the test data

        :param s3_output_file_path_generator_instructions: str
            Path of the generator instruction output for the following modeling steps

        :param s3_output_file_path_modeling: str
            Path of the output files of the following modeling steps

        :param s3_output_file_path_visualization: str
            Path of the output files of the following visualization step

        :param val_data_file_path: str
            Complete file path of the validation data set

        :param algorithm: str
            Abbreviated name of the evolutionary algorithm
                -> ga: Genetic Algorithm
                -> si: Swarm Intelligence (POS)

        :param max_iterations: int
            Maximum number of iterations

        :param pop_size: int
            Size of the population

        :param burn_in_iterations: int
            Number of burn-in iterations

        :param warm_start: bool
            Whether to run with warm start (one individual has standard hyperparameter settings)

        :param change_rate: float
            Rate of the hyperparameter change (mutation / adjustment)

        :param change_prob: float
            Probability of changing hyperparameter (mutation / adjustment)

        :param parents_ratio: float
            Ratio of parenthood

        :param crossover: bool
            Whether to apply crossover inheritance strategy or not (generic algorithm only)

        :param early_stopping: bool
            Whether to enable early stopping or not

        :param convergence: bool
            Whether to enable convergence

        :param convergence_measure: str
            Abbreviated name of the convergence measurement

        :param timer_in_seconds: int
            Timer in seconds for stopping evolution

        :param re_populate: bool
            Whether to re-populate because of poor performance of the entire population or not

        :param re_populate_threshold: float
            Threshold to decide to re-populate

        :param max_trials: int
            Maximum number of trials for re-population

        :param environment_reaction_path: str
            File path of the reaction of the environment to process in each interation

        :param results_table: bool
             Evolution results table
                -> Table Chart

        :param model_evolution: bool
            Evolution of individuals
                -> Scatter Chart

        :param model_distribution: bool
            Distribution of used model types
                -> Bar Chart / Pie Chart

        :param param_distribution: bool
            Distribution of used model parameter combination
                -> Tree Map / Sunburst

        :param train_time_distribution: bool
            Distribution of training time
                -> Violin

        :param breeding_map: bool
            Breeding evolution as
                -> Heat Map

        :param breeding_graph: bool
            Breeding evolution as
                -> Network Graph

        :param fitness_distribution: bool
            Distribution of fitness metric
                -> Ridge Line Chart

        :param fitness_evolution: bool
            Evolution of fitness metric
                -> Line Chart

        :param fitness_dimensions: bool
            Calculated loss value for each dimension in fitness metric
                -> Radar Chart
                -> Tree Map

        :param per_iteration: bool
            Visualize results of each iteration in detail or visualize just evolutionary results

        :param aws_account_id: str
            AWS account id

        :param evolutionary_algorithm_docker_image_name: str
            Name of the docker image repository of the evolutionary algorithm

        :param evolutionary_algorithm_docker_image_tag: str
            Name of the docker image tag of the evolutionary algorithm

        :param evolutionary_algorithm_volume: dsl.VolumeOp
            Attached container volume of the evolutionary algorithm

        :param evolutionary_algorithm_volume_dir: str
            Name of the volume directory of the evolutionary algorithm

        :param evolutionary_algorithm_display_name: str
            Display name of the Kubeflow Pipeline component of the evolutionary algorithm

        :param evolutionary_algorithm_n_cpu_request: str
            Number of requested CPU's of the evolutionary algorithm

        :param evolutionary_algorithm_n_cpu_limit: str
            Maximum number of requested CPU's of the evolutionary algorithm

        :param evolutionary_algorithm_n_gpu: str
            Maximum number of requested GPU's of the evolutionary algorithm

        :param evolutionary_algorithm_gpu_vendor: str
            Name of the GPU vendor of the evolutionary algorithm
                -> amd: AMD
                -> nvidia: NVIDIA

        :param evolutionary_algorithm_memory_request: str
            Memory request of the evolutionary algorithm

        :param evolutionary_algorithm_memory_limit: str
            Limit of the requested memory of the evolutionary algorithm

        :param evolutionary_algorithm_ephemeral_storage_request: str
            Ephemeral storage request (cloud based additional memory storage) of the evolutionary algorithm

        :param evolutionary_algorithm_ephemeral_storage_limit: str
            Limit of the requested ephemeral storage (cloud based additional memory storage) of the evolutionary algorithm

        :param evolutionary_algorithm_instance_name: str
            Name of the used AWS instance (value) of the evolutionary algorithm

        :param evolutionary_algorithm_max_cache_staleness: str
            Maximum of staleness days of the component cache of the evolutionary algorithm
        """
        self.s3_metadata_file_path: str = s3_metadata_file_path
        self.ml_type: str = ml_type
        self.target: str = target
        self.features: Union[List[str], dsl.PipelineParam] = features
        self.models: List[str] = models
        self.metrics: List[str] = metrics
        self.train_data_file_path: str = train_data_file_path
        self.test_data_file_path: str = test_data_file_path
        self.s3_output_file_path_generator_instructions: str = s3_output_file_path_generator_instructions
        self.s3_output_file_path_modeling: str = s3_output_file_path_modeling
        self.s3_output_file_path_visualization: str = s3_output_file_path_evolutionary_algorithm_visualization
        self.s3_output_file_path_evolutionary_algorithm_images: str = s3_output_file_path_evolutionary_algorithm_images
        self.s3_output_file_path_best_model_visualization: str = s3_output_file_path_best_model_visualization
        self.s3_output_file_path_best_model_images: str = s3_output_file_path_best_model_images
        self.val_data_file_path: str = val_data_file_path
        self.algorithm: str = algorithm
        self.max_iterations: int = max_iterations
        self.pop_size: int = pop_size
        self.burn_in_iterations: int = burn_in_iterations
        self.warm_start: bool = warm_start
        self.change_rate: float = change_rate
        self.change_prob: float = change_prob
        self.parents_ratio: float = parents_ratio
        self.crossover: bool = crossover
        self.early_stopping: int = early_stopping
        self.convergence: bool = convergence
        self.convergence_measure: str = convergence_measure
        self.timer_in_seconds: int = timer_in_seconds
        self.re_populate: bool = re_populate
        self.re_populate_threshold: float = re_populate_threshold
        self.max_trials: int = max_trials
        self.environment_reaction_path: str = environment_reaction_path
        self.results_table: bool = results_table
        self.model_distribution: bool = model_distribution
        self.model_evolution: bool = model_evolution
        self.param_distribution: bool = param_distribution
        self.train_time_distribution: bool = train_time_distribution
        self.breeding_map: bool = breeding_map
        self.breeding_graph: bool = breeding_graph
        self.fitness_distribution: bool = fitness_distribution
        self.fitness_evolution: bool = fitness_evolution
        self.fitness_dimensions: bool = fitness_dimensions
        self.per_iteration: bool = per_iteration
        self.aws_account_id: str = aws_account_id
        self.evolutionary_algorithm_docker_image_name: str = evolutionary_algorithm_docker_image_name
        self.evolutionary_algorithm_docker_image_tag: str = evolutionary_algorithm_docker_image_tag
        self.evolutionary_algorithm_volume: dsl.VolumeOp = evolutionary_algorithm_volume
        self.evolutionary_algorithm_volume_dir: str = evolutionary_algorithm_volume_dir
        self.evolutionary_algorithm_display_name: str = evolutionary_algorithm_display_name
        self.evolutionary_algorithm_n_cpu_request: str = evolutionary_algorithm_n_cpu_request
        self.evolutionary_algorithm_n_cpu_limit: str = evolutionary_algorithm_n_cpu_limit
        self.evolutionary_algorithm_n_gpu: str = evolutionary_algorithm_n_gpu
        self.evolutionary_algorithm_gpu_vendor: str = evolutionary_algorithm_gpu_vendor
        self.evolutionary_algorithm_memory_request: str = evolutionary_algorithm_memory_request
        self.evolutionary_algorithm_memory_limit: str = evolutionary_algorithm_memory_limit
        self.evolutionary_algorithm_ephemeral_storage_request: str = evolutionary_algorithm_ephemeral_storage_request
        self.evolutionary_algorithm_ephemeral_storage_limit: str = evolutionary_algorithm_ephemeral_storage_limit
        self.evolutionary_algorithm_instance_name: str = evolutionary_algorithm_instance_name
        self.evolutionary_algorithm_max_cache_staleness: str = evolutionary_algorithm_max_cache_staleness
        self.extract_instruction_python_version: str = extract_instruction_python_version
        self.extract_instruction_display_name: str = extract_instruction_display_name
        self.extract_instruction_n_cpu_request: str = extract_instruction_n_cpu_request
        self.extract_instruction_n_cpu_limit: str = extract_instruction_n_cpu_limit
        self.extract_instruction_n_gpu: str = extract_instruction_n_gpu
        self.extract_instruction_gpu_vendor: str = extract_instruction_gpu_vendor
        self.extract_instruction_memory_request: str = extract_instruction_memory_request
        self.extract_instruction_memory_limit: str = extract_instruction_memory_limit
        self.extract_instruction_ephemeral_storage_request: str = extract_instruction_ephemeral_storage_request
        self.extract_instruction_ephemeral_storage_limit: str = extract_instruction_ephemeral_storage_limit
        self.extract_instruction_instance_name: str = extract_instruction_instance_name
        self.extract_instruction_max_cache_staleness: str = extract_instruction_max_cache_staleness
        self.generate_supervised_model_docker_image_name: str = generate_supervised_model_docker_image_name
        self.generate_supervised_model_docker_image_tag: str = generate_supervised_model_docker_image_tag
        self.generate_supervised_model_volume: dsl.VolumeOp = generate_supervised_model_volume
        self.generate_supervised_model_volume_dir: str = generate_supervised_model_volume_dir
        self.generate_supervised_model_display_name: str = generate_supervised_model_display_name
        self.generate_supervised_model_n_cpu_request: str = generate_supervised_model_n_cpu_request
        self.generate_supervised_model_n_cpu_limit: str = generate_supervised_model_n_cpu_limit
        self.generate_supervised_model_n_gpu: str = generate_supervised_model_n_gpu
        self.generate_supervised_model_gpu_vendor: str = generate_supervised_model_gpu_vendor
        self.generate_supervised_model_memory_request: str = generate_supervised_model_memory_request
        self.generate_supervised_model_memory_limit: str = generate_supervised_model_memory_limit
        self.generate_supervised_model_ephemeral_storage_request: str = generate_supervised_model_ephemeral_storage_request
        self.generate_supervised_model_ephemeral_storage_limit: str = generate_supervised_model_ephemeral_storage_limit
        self.generate_supervised_model_instance_name: str = generate_supervised_model_instance_name
        self.generate_supervised_model_max_cache_staleness: str = generate_supervised_model_max_cache_staleness
        self.evaluate_machine_learning_docker_image_name: str = evaluate_machine_learning_docker_image_name
        self.evaluate_machine_learning_docker_image_tag: str = evaluate_machine_learning_docker_image_tag
        self.evaluate_machine_learning_volume: dsl.VolumeOp = evaluate_machine_learning_volume
        self.evaluate_machine_learning_volume_dir: str = evaluate_machine_learning_volume_dir
        self.evaluate_machine_learning_display_name: str = evaluate_machine_learning_display_name
        self.evaluate_machine_learning_n_cpu_request: str = evaluate_machine_learning_n_cpu_request
        self.evaluate_machine_learning_n_cpu_limit: str = evaluate_machine_learning_n_cpu_limit
        self.evaluate_machine_learning_n_gpu: str = evaluate_machine_learning_n_gpu
        self.evaluate_machine_learning_gpu_vendor: str = evaluate_machine_learning_gpu_vendor
        self.evaluate_machine_learning_memory_request: str = evaluate_machine_learning_memory_request
        self.evaluate_machine_learning_memory_limit: str = evaluate_machine_learning_memory_limit
        self.evaluate_machine_learning_ephemeral_storage_request: str = evaluate_machine_learning_ephemeral_storage_request
        self.evaluate_machine_learning_ephemeral_storage_limit: str = evaluate_machine_learning_ephemeral_storage_limit
        self.evaluate_machine_learning_instance_name: str = evaluate_machine_learning_instance_name
        self.evaluate_machine_learning_max_cache_staleness: str = evaluate_machine_learning_max_cache_staleness
        self.serializer_docker_image_name: str = serializer_docker_image_name
        self.serializer_docker_image_tag: str = serializer_docker_image_tag
        self.serializer_volume: dsl.VolumeOp = serializer_volume
        self.serializer_volume_dir: str = serializer_volume_dir
        self.serializer_display_name: str = serializer_display_name
        self.serializer_n_cpu_request: str = serializer_n_cpu_request
        self.serializer_n_cpu_limit: str = serializer_n_cpu_limit
        self.serializer_n_gpu: str = serializer_n_gpu
        self.serializer_gpu_vendor: str = serializer_gpu_vendor
        self.serializer_memory_request: str = serializer_memory_request
        self.serializer_memory_limit: str = serializer_memory_limit
        self.serializer_ephemeral_storage_request: str = serializer_ephemeral_storage_request
        self.serializer_ephemeral_storage_limit: str = serializer_ephemeral_storage_limit
        self.serializer_instance_name: str = serializer_instance_name
        self.serializer_max_cache_staleness: str = serializer_max_cache_staleness
        self.gather_metadata_python_version: str = gather_metadata_python_version
        self.gather_metadata_display_name: str = gather_metadata_display_name
        self.gather_metadata_n_cpu_request: str = gather_metadata_n_cpu_request
        self.gather_metadata_n_cpu_limit: str = gather_metadata_n_cpu_limit
        self.gather_metadata_n_gpu: str = gather_metadata_n_gpu
        self.gather_metadata_gpu_vendor: str = gather_metadata_gpu_vendor
        self.gather_metadata_memory_request: str = gather_metadata_memory_request
        self.gather_metadata_memory_limit: str = gather_metadata_memory_limit
        self.gather_metadata_ephemeral_storage_request: str = gather_metadata_ephemeral_storage_request
        self.gather_metadata_ephemeral_storage_limit: str = gather_metadata_ephemeral_storage_limit
        self.gather_metadata_instance_name: str = gather_metadata_instance_name
        self.gather_metadata_max_cache_staleness: str = gather_metadata_max_cache_staleness
        self.interactive_visualizer_docker_image_name: str = interactive_visualizer_docker_image_name
        self.interactive_visualizer_docker_image_tag: str = interactive_visualizer_docker_image_tag
        self.interactive_visualizer_volume: dsl.VolumeOp = interactive_visualizer_volume
        self.interactive_visualizer_volume_dir: str = interactive_visualizer_volume_dir
        self.interactive_visualizer_display_name: str = interactive_visualizer_display_name
        self.interactive_visualizer_n_cpu_request: str = interactive_visualizer_n_cpu_request
        self.interactive_visualizer_n_cpu_limit: str = interactive_visualizer_n_cpu_limit
        self.interactive_visualizer_n_gpu: str = interactive_visualizer_n_gpu
        self.interactive_visualizer_gpu_vendor: str = interactive_visualizer_gpu_vendor
        self.interactive_visualizer_memory_request: str = interactive_visualizer_memory_request
        self.interactive_visualizer_memory_limit: str = interactive_visualizer_memory_limit
        self.interactive_visualizer_ephemeral_storage_request: str = interactive_visualizer_ephemeral_storage_request
        self.interactive_visualizer_ephemeral_storage_limit: str = interactive_visualizer_ephemeral_storage_limit
        self.interactive_visualizer_instance_name: str = interactive_visualizer_instance_name
        self.interactive_visualizer_max_cache_staleness: str = interactive_visualizer_max_cache_staleness
        self.gather_metadata_python_version: str = gather_metadata_python_version
        self.gather_metadata_display_name: str = gather_metadata_display_name
        self.gather_metadata_n_cpu_request: str = gather_metadata_n_cpu_request
        self.gather_metadata_n_cpu_limit: str = gather_metadata_n_cpu_limit
        self.gather_metadata_n_gpu: str = gather_metadata_n_gpu
        self.gather_metadata_gpu_vendor: str = gather_metadata_gpu_vendor
        self.gather_metadata_memory_request: str = gather_metadata_memory_request
        self.gather_metadata_memory_limit: str = gather_metadata_memory_limit
        self.gather_metadata_ephemeral_storage_request: str = gather_metadata_ephemeral_storage_request
        self.gather_metadata_ephemeral_storage_limit: str = gather_metadata_ephemeral_storage_limit
        self.gather_metadata_instance_name: str = gather_metadata_instance_name
        self.gather_metadata_max_cache_staleness: str = gather_metadata_max_cache_staleness
        self.display_visualization_python_version: str = display_visualization_python_version
        self.display_visualization_display_name: str = display_visualization_display_name
        self.display_visualization_n_cpu_request: str = display_visualization_n_cpu_request
        self.display_visualization_n_cpu_limit: str = display_visualization_n_cpu_limit
        self.display_visualization_n_gpu: str = display_visualization_n_gpu
        self.display_visualization_gpu_vendor: str = display_visualization_gpu_vendor
        self.display_visualization_memory_request: str = display_visualization_memory_request
        self.display_visualization_memory_limit: str = display_visualization_memory_limit
        self.display_visualization_ephemeral_storage_request: str = display_visualization_ephemeral_storage_request
        self.display_visualization_ephemeral_storage_limit: str = display_visualization_ephemeral_storage_limit
        self.display_visualization_instance_name: str = display_visualization_instance_name
        self.display_visualization_max_cache_staleness: str = display_visualization_max_cache_staleness
        self.display_metric_python_version: str = display_metric_python_version
        self.display_metric_display_name: str = display_metric_display_name
        self.display_metric_n_cpu_request: str = display_metric_n_cpu_request
        self.display_metric_n_cpu_limit: str = display_metric_n_cpu_limit
        self.display_metric_n_gpu: str = display_metric_n_gpu
        self.display_metric_gpu_vendor: str = display_metric_gpu_vendor
        self.display_metric_memory_request: str = display_metric_memory_request
        self.display_metric_memory_limit: str = display_metric_memory_limit
        self.display_metric_ephemeral_storage_request: str = display_metric_ephemeral_storage_request
        self.display_metric_ephemeral_storage_limit: str = display_metric_ephemeral_storage_limit
        self.display_metric_instance_name: str = display_metric_instance_name
        self.display_metric_max_cache_staleness: str = display_metric_max_cache_staleness

    def _display_metrics(self, file_paths: dsl.PipelineParam) -> dsl.ContainerOp:
        """
        Get dsl.ContainerOp of display metrics component

        :param file_paths: dsl.PipelineParam
            Complete file path of the metrics to display

        :return: dsl.ContainerOp
            Container operator for display metrics
        """
        return display_metrics(file_paths=file_paths,
                               metric_types=None,
                               metric_values=None,
                               metric_formats=None,
                               target_feature=None,
                               prediction_feature=None,
                               labels=None,
                               header=None,
                               python_version=self.display_metric_python_version,
                               display_name=self.display_metric_display_name,
                               n_cpu_request=self.display_metric_n_cpu_request,
                               n_cpu_limit=self.display_metric_n_cpu_limit,
                               n_gpu=self.display_metric_n_gpu,
                               gpu_vendor=self.display_metric_gpu_vendor,
                               memory_request=self.display_metric_memory_request,
                               memory_limit=self.display_metric_memory_limit,
                               ephemeral_storage_request=self.display_metric_ephemeral_storage_request,
                               ephemeral_storage_limit=self.display_metric_ephemeral_storage_limit,
                               instance_name=self.display_metric_instance_name,
                               max_cache_staleness=self.display_metric_max_cache_staleness
                               )

    def _display_visualization(self, file_paths: dsl.PipelineParam) -> dsl.ContainerOp:
        """
        Get dsl.ContainerOp of display visualization component

        :param file_paths: dsl.PipelineParam
            Complete file path of the images to display

        :return: dsl.ContainerOp
            Container operator for display visualization
        """
        return display_visualization(file_paths=file_paths,
                                     python_version=self.display_visualization_python_version,
                                     display_name=self.display_visualization_display_name,
                                     n_cpu_request=self.display_visualization_n_cpu_request,
                                     n_cpu_limit=self.display_visualization_n_cpu_limit,
                                     n_gpu=self.display_visualization_n_gpu,
                                     gpu_vendor=self.display_visualization_gpu_vendor,
                                     memory_request=self.display_visualization_memory_request,
                                     memory_limit=self.display_visualization_memory_limit,
                                     ephemeral_storage_request=self.display_visualization_ephemeral_storage_request,
                                     ephemeral_storage_limit=self.display_visualization_ephemeral_storage_limit,
                                     instance_name=self.display_visualization_instance_name,
                                     max_cache_staleness=self.display_visualization_max_cache_staleness
                                     )

    def _iterate(self, idx: dsl.PipelineParam) -> dsl.ContainerOp:
        """
        Iteration of evolutionary algorithm using Kubeflow graph components

        :param idx: dsl.PipelineParam
            Index values used to iterate in parallelized loop

        :return dsl.ContainerOp
            Container operator of last component in iteration
        """
        with dsl.ParallelFor(loop_args=idx) as item:
            _task_2: dsl.ContainerOp = extract_instruction(idx=item,
                                                           generator_instructions_file_path=self.s3_output_file_path_generator_instructions,
                                                           python_version=self.extract_instruction_python_version,
                                                           display_name=self.extract_instruction_display_name,
                                                           n_cpu_request=self.extract_instruction_n_cpu_request,
                                                           n_cpu_limit=self.extract_instruction_n_cpu_limit,
                                                           n_gpu=self.extract_instruction_n_gpu,
                                                           gpu_vendor=self.extract_instruction_gpu_vendor,
                                                           memory_request=self.extract_instruction_memory_request,
                                                           memory_limit=self.extract_instruction_memory_limit,
                                                           ephemeral_storage_request=self.extract_instruction_ephemeral_storage_request,
                                                           ephemeral_storage_limit=self.extract_instruction_ephemeral_storage_limit,
                                                           instance_name=self.extract_instruction_instance_name,
                                                           max_cache_staleness=self.extract_instruction_max_cache_staleness
                                                           )
            _task_3: dsl.ContainerOp = generate_supervised_model(ml_type=self.ml_type,
                                                                 model_name=_task_2.outputs['model_name'],
                                                                 target_feature=self.target,
                                                                 train_data_set_path=self.train_data_file_path,
                                                                 test_data_set_path=self.test_data_file_path,
                                                                 s3_output_path_model=_task_2.outputs['model_artifact_path'],
                                                                 s3_output_path_param=_task_2.outputs['model_param_path'],
                                                                 s3_output_path_metadata=_task_2.outputs['model_metadata_path'],
                                                                 s3_output_path_evaluation_train_data=_task_2.outputs['evaluate_train_data_path'],
                                                                 s3_output_path_evaluation_test_data=_task_2.outputs['evaluate_test_data_path'],
                                                                 predictors=self.features,
                                                                 model_id=_task_2.outputs['id'],
                                                                 model_param_path=_task_2.outputs['model_input_param_path'],
                                                                 param_rate=_task_2.outputs['param_rate'],
                                                                 warm_start=_task_2.outputs['warm_start'],
                                                                 docker_image_name=self.generate_supervised_model_docker_image_name,
                                                                 docker_image_tag=self.generate_supervised_model_docker_image_tag,
                                                                 display_name=self.generate_supervised_model_display_name,
                                                                 n_cpu_request=self.generate_supervised_model_n_cpu_request,
                                                                 n_cpu_limit=self.generate_supervised_model_n_cpu_limit,
                                                                 n_gpu=self.generate_supervised_model_n_gpu,
                                                                 gpu_vendor=self.generate_supervised_model_gpu_vendor,
                                                                 memory_request=self.generate_supervised_model_memory_request,
                                                                 memory_limit=self.generate_supervised_model_memory_limit,
                                                                 ephemeral_storage_request=self.generate_supervised_model_ephemeral_storage_request,
                                                                 ephemeral_storage_limit=self.generate_supervised_model_ephemeral_storage_limit,
                                                                 instance_name=self.generate_supervised_model_instance_name,
                                                                 max_cache_staleness=self.generate_supervised_model_max_cache_staleness
                                                                 )
            _task_4: dsl.ContainerOp = evaluate_machine_learning(ml_type=self.ml_type,
                                                                 target_feature_name=self.target,
                                                                 prediction_feature_name='prediction',
                                                                 train_data_set_path=_task_2.outputs['evaluate_train_data_path'],
                                                                 test_data_set_path=_task_2.outputs['evaluate_test_data_path'],
                                                                 metrics=self.metrics,
                                                                 s3_output_path_metrics=_task_2.outputs['model_fitness_path'],
                                                                 docker_image_name=self.evaluate_machine_learning_docker_image_name,
                                                                 docker_image_tag=self.evaluate_machine_learning_docker_image_tag,
                                                                 display_name=self.evaluate_machine_learning_display_name,
                                                                 n_cpu_request=self.evaluate_machine_learning_n_cpu_request,
                                                                 n_cpu_limit=self.evaluate_machine_learning_n_cpu_limit,
                                                                 n_gpu=self.evaluate_machine_learning_n_gpu,
                                                                 gpu_vendor=self.evaluate_machine_learning_gpu_vendor,
                                                                 memory_request=self.evaluate_machine_learning_memory_request,
                                                                 memory_limit=self.evaluate_machine_learning_memory_limit,
                                                                 ephemeral_storage_request=self.evaluate_machine_learning_ephemeral_storage_request,
                                                                 ephemeral_storage_limit=self.evaluate_machine_learning_ephemeral_storage_limit,
                                                                 instance_name=self.evaluate_machine_learning_instance_name,
                                                                 max_cache_staleness=self.evaluate_machine_learning_max_cache_staleness
                                                                 )
            _task_4.after(_task_3)
        _task_5: dsl.ContainerOp = serializer(action='evolutionary_algorithm',
                                              parallelized_obj=[self.s3_output_file_path_generator_instructions],
                                              s3_output_file_path_parallelized_data=self.environment_reaction_path,
                                              docker_image_name=self.serializer_docker_image_name,
                                              docker_image_tag=self.serializer_docker_image_tag,
                                              display_name=self.serializer_display_name,
                                              n_cpu_request=self.serializer_n_cpu_request,
                                              n_cpu_limit=self.serializer_n_cpu_limit,
                                              n_gpu=self.serializer_n_gpu,
                                              gpu_vendor=self.serializer_gpu_vendor,
                                              memory_request=self.serializer_memory_request,
                                              memory_limit=self.serializer_memory_limit,
                                              ephemeral_storage_request=self.serializer_ephemeral_storage_request,
                                              ephemeral_storage_limit=self.serializer_ephemeral_storage_limit,
                                              instance_name=self.serializer_instance_name,
                                              max_cache_staleness=self.serializer_max_cache_staleness
                                              )
        _task_5.after(_task_4)
        return _task_5

    def _gather_metadata(self) -> dsl.ContainerOp:
        """
        Get dsl.ContainerOp of gather metadata component

        :return: dsl.ContainerOp
            Container operator for gather metadata
        """
        return gather_metadata(metadata_file_path=self.s3_metadata_file_path,
                               modeling_file_path=self.s3_output_file_path_modeling,
                               environment_reaction_file_path=self.environment_reaction_path,
                               python_version=self.gather_metadata_python_version,
                               display_name=self.gather_metadata_display_name,
                               n_cpu_request=self.gather_metadata_n_cpu_request,
                               n_cpu_limit=self.gather_metadata_n_cpu_limit,
                               n_gpu=self.gather_metadata_n_gpu,
                               gpu_vendor=self.gather_metadata_gpu_vendor,
                               memory_request=self.gather_metadata_memory_request,
                               memory_limit=self.gather_metadata_memory_limit,
                               ephemeral_storage_request=self.gather_metadata_ephemeral_storage_request,
                               ephemeral_storage_limit=self.gather_metadata_ephemeral_storage_limit,
                               instance_name=self.gather_metadata_instance_name,
                               max_cache_staleness=self.gather_metadata_max_cache_staleness
                               )

    def _get_evolutionary_algorithm_container_op(self,
                                                 output_file_path_evolve: str = 'evolve.json',
                                                 output_file_path_stopping_reason: str = 'stopping_reason.json',
                                                 output_file_path_individual_idx: str = 'individual_idx.json'
                                                 ) -> dsl.ContainerOp:
        """
        Get dsl.ContainerOp of evolutionary algorithm component

        :return: dsl.ContainerOp
            Container operator for evolutionary algorithm
        """
        _volume: dict = {self.evolutionary_algorithm_volume_dir: self.evolutionary_algorithm_volume if self.evolutionary_algorithm_volume is None else self.evolutionary_algorithm_volume.volume}
        _arguments: list = ['-s3_metadata_file_path', self.s3_metadata_file_path,
                            '-target', self.target,
                            '-features', self.features,
                            '-models', self.models,
                            '-train_data_file_path', self.train_data_file_path,
                            '-test_data_file_path', self.test_data_file_path,
                            '-output_file_path_evolve', output_file_path_evolve,
                            '-output_file_path_stopping_reason', output_file_path_stopping_reason,
                            '-output_file_path_individual_idx', output_file_path_individual_idx,
                            '-s3_output_file_path_generator_instructions', self.s3_output_file_path_generator_instructions,
                            '-s3_output_file_path_modeling', self.s3_output_file_path_modeling,
                            '-s3_output_file_path_visualization', self.s3_output_file_path_visualization,
                            '-algorithm', self.algorithm,
                            '-max_iterations', self.max_iterations,
                            '-pop_size', self.pop_size,
                            '-burn_in_iterations', self.burn_in_iterations,
                            '-warm_start', int(self.warm_start),
                            '-change_rate', self.change_rate,
                            '-change_prob', self.change_prob,
                            '-parents_ratio', self.parents_ratio,
                            '-crossover', int(self.crossover),
                            '-early_stopping', int(self.early_stopping),
                            '-convergence', int(self.convergence),
                            '-convergence_measure', self.convergence_measure,
                            '-timer_in_seconds', self.timer_in_seconds,
                            '-re_populate', int(self.re_populate),
                            '-re_populate_threshold', self.re_populate_threshold,
                            '-max_trials', self.max_trials,
                            '-results_table', int(self.results_table),
                            '-model_distribution', int(self.model_distribution),
                            '-model_evolution', int(self.model_evolution),
                            '-param_distribution', int(self.param_distribution),
                            '-train_time_distribution', int(self.train_time_distribution),
                            '-breeding_map', int(self.breeding_map),
                            '-breeding_graph', int(self.breeding_graph),
                            '-fitness_distribution', int(self.fitness_distribution),
                            '-fitness_evolution', int(self.fitness_evolution),
                            '-fitness_dimensions', int(self.fitness_dimensions),
                            '-per_iteration', int(self.per_iteration)
                            ]
        if self.val_data_file_path is not None:
            _arguments.extend(['-val_data_file_path', self.val_data_file_path])
        if self.environment_reaction_path is not None:
            _arguments.extend(['-environment_reaction_path', self.environment_reaction_path])
        _task: dsl.ContainerOp = dsl.ContainerOp(name='evolutionary_algorithm',
                                                 image=f'{self.aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{self.evolutionary_algorithm_docker_image_name}:{self.evolutionary_algorithm_docker_image_tag}',
                                                 command=["python", "task.py"],
                                                 arguments=_arguments,
                                                 init_containers=None,
                                                 sidecars=None,
                                                 container_kwargs=None,
                                                 artifact_argument_paths=None,
                                                 file_outputs={'evolve': output_file_path_evolve,
                                                               'stopping_reason': output_file_path_stopping_reason,
                                                               'idx': output_file_path_individual_idx
                                                               },
                                                 output_artifact_paths=None,
                                                 is_exit_handler=False,
                                                 pvolumes=self.evolutionary_algorithm_volume if self.evolutionary_algorithm_volume is None else _volume
                                                 )
        _task.set_display_name(self.evolutionary_algorithm_display_name)
        add_container_op_parameters(container_op=_task,
                                    n_cpu_request=self.evolutionary_algorithm_n_cpu_request,
                                    n_cpu_limit=self.evolutionary_algorithm_n_cpu_limit,
                                    n_gpu=self.evolutionary_algorithm_n_gpu,
                                    gpu_vendor=self.evolutionary_algorithm_gpu_vendor,
                                    memory_request=self.evolutionary_algorithm_memory_request,
                                    memory_limit=self.evolutionary_algorithm_memory_limit,
                                    ephemeral_storage_request=self.evolutionary_algorithm_ephemeral_storage_request,
                                    ephemeral_storage_limit=self.evolutionary_algorithm_ephemeral_storage_limit,
                                    instance_name=self.evolutionary_algorithm_instance_name,
                                    max_cache_staleness=self.evolutionary_algorithm_max_cache_staleness
                                    )
        return _task

    def _interactive_visualizer(self) -> dsl.ContainerOp:
        """
        Get dsl.ContainerOp of interactive visualization component

        :return: dsl.ContainerOp
            Container operator for interactive visualization
        """
        return interactive_visualizer(s3_output_image_path='',
                                      data_set_path=None,
                                      analytical_data_types_path='',
                                      subplots_file_path=self.s3_output_file_path_visualization,
                                      docker_image_name=self.interactive_visualizer_docker_image_name,
                                      docker_image_tag=self.interactive_visualizer_docker_image_tag,
                                      display_name=self.interactive_visualizer_display_name,
                                      n_cpu_request=self.interactive_visualizer_n_cpu_request,
                                      n_cpu_limit=self.interactive_visualizer_n_cpu_limit,
                                      n_gpu=self.interactive_visualizer_n_gpu,
                                      gpu_vendor=self.interactive_visualizer_gpu_vendor,
                                      memory_request=self.interactive_visualizer_memory_request,
                                      memory_limit=self.interactive_visualizer_memory_limit,
                                      ephemeral_storage_request=self.interactive_visualizer_ephemeral_storage_request,
                                      ephemeral_storage_limit=self.interactive_visualizer_memory_limit,
                                      instance_name=self.interactive_visualizer_instance_name,
                                      max_cache_staleness=self.interactive_visualizer_max_cache_staleness
                                      )

    def hyperparameter_tuning(self) -> Tuple[dsl.ContainerOp, dsl.ContainerOp]:
        """
        Run hyperparameter tuning using evolutionary algorithm Kubeflow graph components

        :return: Tuple[dsl.ContainerOp, dsl.ContainerOp]
            Container operator of the first and the last component after complete iteration
        """
        _task_0: dsl.ContainerOp = self._get_evolutionary_algorithm_container_op()
        _task_1: dsl.ContainerOp = self._iterate(idx=_task_0.outputs['idx'])
        with dsl.Condition(condition=_task_0.outputs['evolve'] == 1, name='Stop-Evolution-Layer-0'):
            _task_x: dsl.ContainerOp = self._gather_metadata()
            _task_x.after(_task_1)
        with dsl.Condition(condition=_task_0.outputs['evolve'] == 0, name='Stop-Evolution-Layer-1'):
            _task_2: dsl.ContainerOp = self._gather_metadata()
            _task_2.after(_task_1)
            _task_3: dsl.ContainerOp = self._interactive_visualizer()
            _task_3.after(_task_2)
        return _task_0, _task_x


def evolutionary_algorithm(s3_metadata_file_path: Union[str, dsl.PipelineParam],
                           target: Union[str, dsl.PipelineParam],
                           features: Union[List[str], dsl.PipelineParam],
                           models: Union[List[str], dsl.PipelineParam],
                           train_data_file_path: Union[str, dsl.PipelineParam],
                           test_data_file_path: Union[str, dsl.PipelineParam],
                           s3_output_file_path_generator_instructions: Union[str, dsl.PipelineParam],
                           s3_output_file_path_modeling: Union[str, dsl.PipelineParam],
                           s3_output_file_path_visualization: Union[str, dsl.PipelineParam],
                           output_file_path_evolve: str = 'evolve.json',
                           output_file_path_stopping_reason: str = 'stopping_reason.json',
                           output_file_path_individual_idx: str = 'individual_idx.json',
                           val_data_file_path: str = None,
                           algorithm: str = 'ga',
                           max_iterations: int = 10,
                           pop_size: int = 64,
                           burn_in_iterations: int = -1,
                           warm_start: bool = True,
                           change_rate: float = 0.1,
                           change_prob: float = 0.85,
                           parents_ratio: float = 0.5,
                           crossover: bool = True,
                           early_stopping: int = 0,
                           convergence: bool = False,
                           convergence_measure: str = 'min',
                           timer_in_seconds: int = 43200,
                           re_populate: bool = False,
                           re_populate_threshold: float = 3.0,
                           max_trials: int = 2,
                           environment_reaction_path: str = None,
                           results_table: bool = True,
                           model_distribution: bool = True,
                           model_evolution: bool = True,
                           param_distribution: bool = False,
                           train_time_distribution: bool = True,
                           breeding_map: bool = False,
                           breeding_graph: bool = False,
                           fitness_distribution: bool = True,
                           fitness_evolution: bool = True,
                           fitness_dimensions: bool = True,
                           per_iteration: bool = True,
                           aws_account_id: str = '711117404296',
                           docker_image_name: str = 'ml-ops-evolutionary-algorithm',
                           docker_image_tag: str = 'v1',
                           volume: dsl.VolumeOp = None,
                           volume_dir: str = '/mnt',
                           display_name: str = 'Evolutionary Algorithm',
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
    Optimize machine learning models

    :param s3_metadata_file_path: str
        Complete file path of the metadata

    :param target: str
        Name of the target feature

    :param features: List[str]
        Name of the features

    :param models: List[str]
        Abbreviated name of the machine learning models

    :param train_data_file_path: str
        Complete file path of the training data

    :param test_data_file_path: str
        Complete file path of the test data

    :param s3_output_file_path_generator_instructions: str
        Path of the generator instruction output for the following modeling steps

    :param s3_output_file_path_modeling: str
        Path of the output files of the following modeling steps

    :param s3_output_file_path_visualization: str
        Path of the output files of the following visualization step

    :param output_file_path_evolve: str
        File path of the evolution status output

    :param output_file_path_stopping_reason: str
        File path of the stopping reason output

    :param output_file_path_individual_idx: str
        File path of the individual index of the instruction list to proceed output

    :param val_data_file_path: str
        Complete file path of the validation data set

    :param algorithm: str
        Abbreviated name of the evolutionary algorithm
            -> ga: Genetic Algorithm
            -> si: Swarm Intelligence (POS)

    :param max_iterations: int
        Maximum number of iterations

    :param pop_size: int
        Size of the population

    :param burn_in_iterations: int
        Number of burn-in iterations

    :param warm_start: bool
        Whether to run with warm start (one individual has standard hyperparameter settings)

    :param change_rate: float
        Rate of the hyperparameter change (mutation / adjustment)

    :param change_prob: float
        Probability of changing hyperparameter (mutation / adjustment)

    :param parents_ratio: float
        Ratio of parenthood

    :param crossover: bool
        Whether to apply crossover inheritance strategy or not (generic algorithm only)

    :param early_stopping: bool
        Whether to enable early stopping or not

    :param convergence: bool
        Whether to enable convergence

    :param convergence_measure: str
        Abbreviated name of the convergence measurement

    :param timer_in_seconds: int
        Timer in seconds for stopping evolution

    :param re_populate: bool
        Whether to re-populate because of poor performance of the entire population or not

    :param re_populate_threshold: float
        Threshold to decide to re-populate

    :param max_trials: int
        Maximum number of trials for re-population

    :param environment_reaction_path: str
        File path of the reaction of the environment to process in each interation

    :param results_table: bool
         Evolution results table
            -> Table Chart

    :param model_evolution: bool
        Evolution of individuals
            -> Scatter Chart

    :param model_distribution: bool
        Distribution of used model types
            -> Bar Chart / Pie Chart

    :param param_distribution: bool
        Distribution of used model parameter combination
            -> Tree Map / Sunburst

    :param train_time_distribution: bool
        Distribution of training time
            -> Violin

    :param breeding_map: bool
        Breeding evolution as
            -> Heat Map

    :param breeding_graph: bool
        Breeding evolution as
            -> Network Graph

    :param fitness_distribution: bool
        Distribution of fitness metric
            -> Ridge Line Chart

    :param fitness_evolution: bool
        Evolution of fitness metric
            -> Line Chart

    :param fitness_dimensions: bool
        Calculated loss value for each dimension in fitness metric
            -> Radar Chart
            -> Tree Map

    :param per_iteration: bool
        Visualize results of each iteration in detail or visualize just evolutionary results

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
        Container operator for evolutionary algorithm
    """
    _volume: dict = {volume_dir: volume if volume is None else volume.volume}
    _arguments: list = ['-s3_metadata_file_path', s3_metadata_file_path,
                        '-target', target,
                        '-features', features,
                        '-models', models,
                        '-train_data_file_path', train_data_file_path,
                        '-test_data_file_path', test_data_file_path,
                        '-output_file_path_evolve', output_file_path_evolve,
                        '-output_file_path_stopping_reason', output_file_path_stopping_reason,
                        '-output_file_path_individual_idx', output_file_path_individual_idx,
                        '-s3_output_file_path_generator_instructions', s3_output_file_path_generator_instructions,
                        '-s3_output_file_path_modeling', s3_output_file_path_modeling,
                        '-s3_output_file_path_visualization', s3_output_file_path_visualization,
                        '-algorithm', algorithm,
                        '-max_iterations', max_iterations,
                        '-pop_size', pop_size,
                        '-burn_in_iterations', burn_in_iterations,
                        '-warm_start', int(warm_start),
                        '-change_rate', change_rate,
                        '-change_prob', change_prob,
                        '-parents_ratio', parents_ratio,
                        '-crossover', int(crossover),
                        '-early_stopping', int(early_stopping),
                        '-convergence', int(convergence),
                        '-convergence_measure', convergence_measure,
                        '-timer_in_seconds', timer_in_seconds,
                        '-re_populate', int(re_populate),
                        '-re_populate_threshold', re_populate_threshold,
                        '-max_trials', max_trials,
                        '-results_table', int(results_table),
                        '-model_distribution', int(model_distribution),
                        '-model_evolution', int(model_evolution),
                        '-param_distribution', int(param_distribution),
                        '-train_time_distribution', int(train_time_distribution),
                        '-breeding_map', int(breeding_map),
                        '-breeding_graph', int(breeding_graph),
                        '-fitness_distribution', int(fitness_distribution),
                        '-fitness_evolution', int(fitness_evolution),
                        '-fitness_dimensions', int(fitness_dimensions),
                        '-per_iteration', int(per_iteration)
                        ]
    if val_data_file_path is not None:
        _arguments.extend(['-val_data_file_path', val_data_file_path])
    if environment_reaction_path is not None:
        _arguments.extend(['-environment_reaction_path', environment_reaction_path])
    _task: dsl.ContainerOp = dsl.ContainerOp(name='evolutionary_algorithm',
                                             image=f'{aws_account_id}.dkr.ecr.eu-central-1.amazonaws.com/{docker_image_name}:{docker_image_tag}',
                                             command=["python", "task.py"],
                                             arguments=_arguments,
                                             init_containers=None,
                                             sidecars=None,
                                             container_kwargs=None,
                                             artifact_argument_paths=None,
                                             file_outputs={'evolve': output_file_path_evolve,
                                                           'stopping_reason': output_file_path_stopping_reason,
                                                           'idx': output_file_path_individual_idx
                                                           },
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


def _extract(idx: Union[int, dsl.PipelineParam],
             generator_instructions_file_path: str
             ) -> NamedTuple('outputs', [('idx', int),
                                         ('id', int),
                                         ('parent', int),
                                         ('model_name', str),
                                         ('params', dict),
                                         ('param_rate', float),
                                         ('warm_start', int),
                                         ('model_artifact_path', str),
                                         ('model_input_param_path', str),
                                         ('model_param_path', str),
                                         ('model_metadata_path', str),
                                         ('model_fitness_path', str),
                                         ('evaluate_train_data_path', str),
                                         ('evaluate_test_data_path', str),
                                         ('evaluate_val_data_path', str)
                                         ]
                             ):
    """
    Extract model generator instructions generated by evolutionary algorithm

    :param idx: int
        Index value of the generator instruction to extract

    :param generator_instructions_file_path: str
        Complete file path of the generator instructions

    :return NamedTuple
        Extracted elements from instructions
    """
    import boto3
    import json
    _complete_file_path: str = generator_instructions_file_path.replace('s3://', '')
    _bucket_name: str = _complete_file_path.split('/')[0]
    _file_path: str = _complete_file_path.replace(f'{_bucket_name}/', '')
    _file_type: str = _complete_file_path.split('.')[-1]
    _s3_resource: boto3 = boto3.resource('s3')
    _obj: bytes = _s3_resource.Bucket(_bucket_name).Object(_file_path).get()['Body'].read()
    _generator_instructions: dict = json.loads(_obj)[int(idx)]
    _idx: int = _generator_instructions.get('idx')
    _id: int = _generator_instructions.get('id')
    _parent: int = _generator_instructions.get('parent')
    _model_name: str = _generator_instructions.get('model_name')
    _params: dict = {} if _generator_instructions.get('params') is None else _generator_instructions.get('params')
    _param_rate: float = _generator_instructions.get('param_rate')
    _warm_start: int = _generator_instructions.get('warm_start')
    _model_artifact_path: str = _generator_instructions.get('model_artifact_path')
    _model_input_param_path: str = '' if _generator_instructions.get('model_input_param_path') is None else _generator_instructions.get('model_input_param_path')
    _model_param_path: str = _generator_instructions.get('model_param_path')
    _model_metadata_path: str = _generator_instructions.get('model_metadata_path')
    _model_fitness_path: str = _generator_instructions.get('model_fitness_path')
    _evaluate_train_data_path: str = _generator_instructions.get('evaluate_train_data_path')
    _evaluate_test_data_path: str = _generator_instructions.get('evaluate_test_data_path')
    _evaluate_val_data_path: str = _generator_instructions.get('evaluate_val_data_path')
    return [_idx,
            _id,
            _parent,
            _model_name,
            _params,
            _param_rate,
            _warm_start,
            _model_artifact_path,
            _model_input_param_path,
            _model_param_path,
            _model_metadata_path,
            _model_fitness_path,
            _evaluate_train_data_path,
            _evaluate_test_data_path,
            _evaluate_val_data_path
            ]


def extract_instruction(idx: Union[int, dsl.PipelineParam],
                        generator_instructions_file_path: str,
                        python_version: str = '3.9',
                        display_name: str = 'Extract Instruction',
                        n_cpu_request: str = None,
                        n_cpu_limit: str = None,
                        n_gpu: str = None,
                        gpu_vendor: str = 'nvidia',
                        memory_request: str = '1G',
                        memory_limit: str = None,
                        ephemeral_storage_request: str = '1G',
                        ephemeral_storage_limit: str = None,
                        instance_name: str = 'm5.xlarge',
                        max_cache_staleness: str = 'P0D'
                        ) -> dsl.ContainerOp:
    """
    Extract model generator instructions generated by evolutionary algorithm

    :param idx: Any
        Index value of the generator instruction to extract

    :param generator_instructions_file_path: str
        Complete file path of the generator instructions

    :param python_version: str
        Python version of the base image

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
        Container operator for evolutionary algorithm extract instruction
    """
    _container_from_func: dsl.component = create_component_from_func(func=_extract,
                                                                     output_component_file=None,
                                                                     base_image=f'python:{python_version}',
                                                                     packages_to_install=['boto3==1.34.11'],
                                                                     annotations=None
                                                                     )
    _task: dsl.ContainerOp = _container_from_func(idx=idx, generator_instructions_file_path=generator_instructions_file_path)
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


def _gather_metadata(metadata_file_path: str,
                     modeling_file_path: str,
                     environment_reaction_file_path: str
                     ) -> NamedTuple('outputs', [('idx', int),
                                                 ('id', int),
                                                 ('model_name', str),
                                                 ('params', dict),
                                                 ('param_changed', dict),
                                                 ('fitness_metric', float),
                                                 ('fitness_score', float),
                                                 ('model_artifact_path', str),
                                                 ('evaluation_train_data_file_path', str),
                                                 ('evaluation_test_data_file_path', str),
                                                 ('evaluation_val_data_file_path', str)
                                                 ]
                                     ):
    """
    Gather evolution metadata

    :param metadata_file_path: str
        Complete file path of the metadata of the evolutionary algorithm

    :param environment_reaction_file_path: str
        File path of the reaction of the environment to process in each interation

    :return NamedTuple
        Extracted elements of best model generated by evolutionary algorithm
    """
    import boto3
    import copy
    import json
    import numpy as np
    import os
    from datetime import datetime
    _logger_time: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _s3_resource: boto3 = boto3.resource('s3')
    # load metadata file from AWS S3:
    _complete_file_path_metadata: str = metadata_file_path.replace('s3://', '')
    _bucket_name_metadata: str = _complete_file_path_metadata.split('/')[0]
    _file_path_metadata: str = _complete_file_path_metadata.replace(f'{_bucket_name_metadata}/', '')
    _file_type_metadata: str = _complete_file_path_metadata.split('.')[-1]
    _obj_metadata: bytes = _s3_resource.Bucket(_bucket_name_metadata).Object(_file_path_metadata).get()['Body'].read()
    _metadata: dict = json.loads(_obj_metadata)
    # load environment reaction file from AWS S3:
    _complete_file_path_env_reaction: str = environment_reaction_file_path.replace('s3://', '')
    _bucket_name_env_reaction: str = _complete_file_path_env_reaction.split('/')[0]
    _file_path_env_reaction: str = _complete_file_path_env_reaction.replace(f'{_bucket_name_env_reaction}/', '')
    _file_type_env_reaction: str = _complete_file_path_env_reaction.split('.')[-1]
    _obj_env_reaction: bytes = _s3_resource.Bucket(_bucket_name_env_reaction).Object(_file_path_env_reaction).get()['Body'].read()
    _env_reaction: dict = json.loads(_obj_env_reaction)
    # gather metadata of evolutionary process:
    _metadata['iteration_history']['time'].append((datetime.now() - datetime.strptime(_metadata['start_time'][-1], '%Y-%m-%d %H:%M:%S')).seconds)
    if _metadata['iteration_history']['population'].get(f'iter_{_metadata["current_iteration"]}') is None:
        _metadata['iteration_history']['population'].update({f'iter_{_metadata["current_iteration"]}': dict(id=[],
                                                                                                            model_name=[],
                                                                                                            parent=[],
                                                                                                            fitness=[]
                                                                                                            )
                                                             })
    if _metadata['current_iteration'] == 0:
        _fittest_individual_previous_iteration: List[int] = []
    else:
        if _metadata['current_iteration_algorithm'][-1] == 'ga':
            _fittest_individual_previous_iteration: List[int] = _metadata['parents_idx']
        else:
            _fittest_individual_previous_iteration: List[int] = [_metadata['best_global_idx'][-1], _metadata['best_local_idx'][-1]]
    for i in range(0, _metadata['pop_size'], 1):
        if i not in _fittest_individual_previous_iteration:
            # current iteration metadata:
            if _metadata["current_iteration"] == 0:
                _metadata['current_iteration_meta_data']['id'].append(_env_reaction[str(i)]['id'])
                _metadata['current_iteration_meta_data']['model_name'].append(_env_reaction[str(i)]['model_name'])
                _metadata['current_iteration_meta_data']['param'].append(_env_reaction[str(i)]['param'])
                _metadata['current_iteration_meta_data']['param_changed'].append(_env_reaction[str(i)]['param_changed'])
                _metadata['current_iteration_meta_data']['fitness_metric'].append(_env_reaction[str(i)]['fitness_metric'])
                _metadata['current_iteration_meta_data']['fitness_score'].append(_env_reaction[str(i)]['fitness_score'])
            else:
                _metadata['current_iteration_meta_data']['id'][i] = copy.deepcopy(_env_reaction[str(i)]['id'])
                _metadata['current_iteration_meta_data']['model_name'][i] = copy.deepcopy(_env_reaction[str(i)]['model_name'])
                _metadata['current_iteration_meta_data']['param'][i] = copy.deepcopy(_env_reaction[str(i)]['param'])
                _metadata['current_iteration_meta_data']['param_changed'][i] = copy.deepcopy(_env_reaction[str(i)]['param_changed'])
                _metadata['current_iteration_meta_data']['fitness_metric'][i] = copy.deepcopy(_env_reaction[str(i)]['fitness_metric'])
                _metadata['current_iteration_meta_data']['fitness_score'][i] = copy.deepcopy(_env_reaction[str(i)]['fitness_score'])
            print(f'{_logger_time} Fitness score {_env_reaction[str(i)]["fitness_score"]} of individual {i}')
            print(f'{_logger_time} Fitness metric {_env_reaction[str(i)]["fitness_metric"]} of individual {i}')
            # iteration history:
            _metadata['iteration_history']['population'][f'iter_{_metadata["current_iteration"]}']['id'].append(_env_reaction[str(i)]['id'])
            _metadata['iteration_history']['population'][f'iter_{_metadata["current_iteration"]}']['model_name'].append(_env_reaction[str(i)]['model_name'])
            _metadata['iteration_history']['population'][f'iter_{_metadata["current_iteration"]}']['parent'].append(_env_reaction[str(i)]['parent'])
            _metadata['iteration_history']['population'][f'iter_{_metadata["current_iteration"]}']['fitness'].append(_env_reaction[str(i)]['fitness_score'])
            # evolution history:
            _metadata['evolution_history']['id'].append(_env_reaction[str(i)]['id'])
            _metadata['evolution_history']['iteration'].append(_metadata['current_iteration'])
            _metadata['evolution_history']['model_name'].append(_env_reaction[str(i)]['model_name'])
            _metadata['evolution_history']['parent'].append(_env_reaction[str(i)]['parent'])
            _metadata['evolution_history']['change_type'].append(_env_reaction[str(i)]['change_type'])
            _metadata['evolution_history']['fitness_score'].append(_env_reaction[str(i)]['fitness_score'])
            _metadata['evolution_history']['ml_metric'].append(_env_reaction[str(i)]['fitness_metric'])
            _metadata['evolution_history']['train_test_diff'].append(_env_reaction[str(i)]['train_test_diff'])
            _metadata['evolution_history']['train_time_in_seconds'].append(_env_reaction[str(i)]['train_time_in_seconds'])
            _metadata['evolution_history']['original_ml_train_metric'].append(_env_reaction[str(i)]['original_ml_train_metric'])
            _metadata['evolution_history']['original_ml_test_metric'].append(_env_reaction[str(i)]['original_ml_test_metric'])
    # evolution gradient:
    _current_iteration_fitness_scores: List[float] = _metadata['current_iteration_meta_data']['fitness_score']
    _metadata['evolution_gradient']['min'].append(copy.deepcopy(min(_current_iteration_fitness_scores)))
    _metadata['evolution_gradient']['median'].append(copy.deepcopy(np.median(_current_iteration_fitness_scores)))
    _metadata['evolution_gradient']['mean'].append(copy.deepcopy(np.mean(_current_iteration_fitness_scores)))
    _metadata['evolution_gradient']['max'].append(copy.deepcopy(max(_current_iteration_fitness_scores)))
    print(f'{_logger_time} Fitness: Max    -> {_metadata["evolution_gradient"].get("max")[-1]}')
    print(f'{_logger_time} Fitness: Median -> {_metadata["evolution_gradient"].get("median")[-1]}')
    print(f'{_logger_time} Fitness: Mean   -> {_metadata["evolution_gradient"].get("mean")[-1]}')
    print(f'{_logger_time} Fitness: Min    -> {_metadata["evolution_gradient"].get("min")[-1]}')
    # save gathered metadata to AWS S3:
    _s3_client: boto3.client = boto3.client('s3')
    _s3_client.put_object(Body=json.dumps(obj=_metadata), Bucket=_bucket_name_metadata, Key=_file_path_metadata)
    print(f'{_logger_time} Save metadata file: {metadata_file_path}')
    # retrieve currently evolved best model:
    _idx: int = np.array(_metadata['current_iteration_meta_data']['fitness_score']).argmax().item()
    _id: int = _metadata['current_iteration_meta_data']['id'][_idx]
    _model_name: str = _metadata['current_iteration_meta_data']['model_name'][_idx]
    _params: dict = _metadata['current_iteration_meta_data']['param'][_idx]
    _param_changed: dict = _metadata['current_iteration_meta_data']['param_changed'][_idx]
    _fitness_metric: float = _metadata['current_iteration_meta_data']['fitness_metric'][_idx]
    _fitness_score: float = _metadata['current_iteration_meta_data']['fitness_score'][_idx]
    # retrieve model related data from generator instructions file:
    _model_artifact_path: str = os.path.join(modeling_file_path, f'model_artifact_{_id}.joblib')
    _model_param_path: str = os.path.join(modeling_file_path, f'model_param_{_id}.json')
    _model_metadata_path: str = os.path.join(modeling_file_path, f'model_metadata_{_id}.json')
    _evaluate_train_data_path: str = os.path.join(modeling_file_path, f'evaluate_train_data_{_id}.json')
    _evaluate_test_data_path: str = os.path.join(modeling_file_path, f'evaluate_test_data_{_id}.json')
    _evaluate_val_data_path: str = os.path.join(modeling_file_path, f'evaluate_val_data_{_id}.json')
    return [_idx,
            _id,
            _model_name,
            _params,
            _param_changed,
            _fitness_metric,
            _fitness_score,
            _model_artifact_path,
            _evaluate_train_data_path,
            _evaluate_test_data_path,
            _evaluate_val_data_path
            ]


def gather_metadata(metadata_file_path: str,
                    modeling_file_path: str,
                    environment_reaction_file_path: str,
                    python_version: str = '3.9',
                    display_name: str = 'Gather Evolution Metadata',
                    n_cpu_request: str = None,
                    n_cpu_limit: str = None,
                    n_gpu: str = None,
                    gpu_vendor: str = 'nvidia',
                    memory_request: str = '100Mi',
                    memory_limit: str = None,
                    ephemeral_storage_request: str = '100Mi',
                    ephemeral_storage_limit: str = None,
                    instance_name: str = 'm5.xlarge',
                    max_cache_staleness: str = 'P0D'
                    ) -> dsl.ContainerOp:
    """
    Gather evolution metadata

    :param metadata_file_path: str
        Complete file path of the metadata of the evolutionary algorithm

    :param modeling_file_path: str
        File path of the model artifact

    :param environment_reaction_file_path: str
        File path of the reaction of the environment to process in each interation

    :param python_version: str
        Python version of the base image

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
        Container operator for evolutionary algorithm extract instruction
    """
    _container_from_func: dsl.component = create_component_from_func(func=_gather_metadata,
                                                                     output_component_file=None,
                                                                     base_image=f'python:{python_version}',
                                                                     packages_to_install=['boto3==1.34.11',
                                                                                          'numpy==1.26.4'
                                                                                          ],
                                                                     annotations=None
                                                                     )
    _task: dsl.ContainerOp = _container_from_func(metadata_file_path=metadata_file_path,
                                                  modeling_file_path=modeling_file_path,
                                                  environment_reaction_file_path=environment_reaction_file_path
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
