"""

Unsupervised (clustering) machine learning algorithms

"""

import copy
import numpy as np
import os
import pandas as pd

from custom_logger import Log
from datetime import datetime
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.xmeans import splitting_type, xmeans
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, KMeans, MeanShift, OPTICS, SpectralClustering
from sklearn.decomposition import FactorAnalysis, FastICA, PCA, NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, kneighbors_graph
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from typing import Dict, List, Tuple

CLUSTERING_ALGORITHMS: Dict[str, str] = dict(apro='affinity_propagation',
                                             agglo='agglomerative_clustering',
                                             birch='birch',
                                             dbscan='dbscan',
                                             factor='factor_analysis',
                                             fagglo='feature_agglomeration',
                                             ica='independent_component_analysis',
                                             isomap='isometric_mapping',
                                             kmeans='kmeans',
                                             kmedians='kmedians',
                                             kmedoids='kmedoids',
                                             lda='latent_dirichlet_allocation',
                                             lle='local_linear_embedding',
                                             mds='multi_dimensional_scaling',
                                             nmf='non_negative_matrix_factorization',
                                             optics='optics',
                                             pca='principal_component_analysis',
                                             spc='spectral_cluster',
                                             spe='spectral_embedding',
                                             tsne='t_distributed_stochastic_neighbor_embedding',
                                             tsvd='truncated_single_value_decomp',
                                             xmeans='xmeans'
                                             )

SPECIAL_PARAMS: Dict[str, str] = dict(kmedians='initial_medians',
                                      kmedoids='initial_index_medoids',
                                      xmeans='initial_centers',
                                      )

CLUSTER_TYPES: Dict[str, List[str]] = dict(partition=['kmeans', 'kmedians', 'kmedoids', 'xmeans'],
                                           hierarchical=['agglo'],
                                           manigfold=['spc', 'spe', 'isomap', 'lle', 'mds', 'tsne'],
                                           component=['factor', 'ica', 'pca']
                                           )


class Clustering:
    """
    Class for handling clustering or dimensionality reduction algorithms
    """
    def __init__(self, cluster_params: dict = None, seed: int = 1234, **kwargs):
        """
        :param cluster_params: dict
            Clustering hyperparameters

        :param seed: int
            Seed
        """
        self.cluster_params: dict = {} if cluster_params is None else cluster_params
        self.seed: int = 1234 if seed <= 0 else seed
        self.kwargs: dict = kwargs

    def affinity_propagation(self) -> AffinityPropagation:
        """
        Config affinity propagation

        :return: AffinityPropagation
            Sklearn object containing the affinity propagation configuration
        """
        return AffinityPropagation(damping=0.5 if self.cluster_params.get('damping') is None else self.cluster_params.get('damping'),
                                   max_iter=200 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                                   convergence_iter=15 if self.cluster_params.get('convergence_iter') is None else self.cluster_params.get('convergence_iter'),
                                   copy=True if self.cluster_params.get('copy') is None else self.cluster_params.get('copy'),
                                   preference=self.cluster_params.get('preference'),
                                   affinity='euclidean' if self.cluster_params.get('affinity') is None else self.cluster_params.get('affinity'),
                                   verbose=False if self.cluster_params.get('verbose') is None else self.cluster_params.get('verbose')
                                   )

    def affinity_propagation_param(self) -> dict:
        """
        Generate Affinity Propagation clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(damping=np.random.uniform(low=0.5 if self.kwargs.get('damping_low') is None else self.kwargs.get('damping_low'),
                                              high=1.0 if self.kwargs.get('damping_high') is None else self.kwargs.get('damping_high')
                                              ),
                    max_iter=np.random.randint(low=5 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    convergence=np.random.randint(low=2 if self.kwargs.get('convergence_low') is None else self.kwargs.get('convergence_low'),
                                                  high=50 if self.kwargs.get('convergence_high') is None else self.kwargs.get('convergence_high')
                                                  ),
                    affinity=np.random.choice(a=['euclidean', 'precomputed'] if self.kwargs.get('affinity') is None else self.kwargs.get('affinity'))
                    )

    def agglomerative_clustering(self) -> AgglomerativeClustering:
        """
        Config agglomerative clustering

        :return: AgglomerativeClustering
            Sklearn object containing the agglomerative clustering configuration
        """
        return AgglomerativeClustering(n_clusters=3 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                                       metric='euclidean' if self.cluster_params.get('affinity') is None else self.cluster_params.get('affinity'),
                                       compute_full_tree='auto' if self.cluster_params.get('compute_full_tree') is None else self.cluster_params.get('compute_full_tree'),
                                       connectivity=self.cluster_params.get('connectivity'),
                                       distance_threshold=self.cluster_params.get('distance_threshold'),
                                       linkage='ward' if self.cluster_params.get('linkage') is None else self.cluster_params.get('linkage'),
                                       memory=self.cluster_params.get('memory')
                                       )

    def agglomerative_clustering_param(self) -> dict:
        """
        Generate Agglomerative clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    metric=np.random.choice(a=['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    linkage=np.random.choice(a=['ward', 'complete', 'average', 'single'] if self.kwargs.get('linkage') is None else self.kwargs.get('linkage'))
                    )

    def birch(self) -> Birch:
        """
        Config birch clustering

        :return: Birch
            Sklearn object containing the birch clustering configuration
        """
        return Birch(threshold=0.5 if self.cluster_params.get('threshold') is None else self.cluster_params.get('threshold'),
                     branching_factor=50 if self.cluster_params.get('branching_factor') is None else self.cluster_params.get('branching_factor'),
                     n_clusters=3 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                     compute_labels=True if self.cluster_params.get('compute_labels') is None else self.cluster_params.get('compute_labels'),
                     copy=True if self.cluster_params.get('copy') is None else self.cluster_params.get('copy'),
                     )

    def birch_param(self) -> dict:
        """
        Generate Birch clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    threshold=np.random.uniform(low=0.1 if self.kwargs.get('threshold_low') is None else self.kwargs.get('threshold_low'),
                                                high=1.0 if self.kwargs.get('threshold_high') is None else self.kwargs.get('threshold_high')
                                                ),
                    branching_factor=np.random.randint(low=5 if self.kwargs.get('branching_factor_low') is None else self.kwargs.get('branching_factor_low'),
                                                       high=100 if self.kwargs.get('branching_factor_high') is None else self.kwargs.get('branching_factor_high')
                                                       )
                    )

    def dbscan(self) -> DBSCAN:
        """
        Config density-based algorithm for discovering clusters in large spatial databases with noise

        :return: DBSCAN
            Sklearn object containing the dbscan clustering configuration
        """
        return DBSCAN(eps=0.5 if self.cluster_params.get('eps') is None else self.cluster_params.get('eps'),
                      min_samples=5 if self.cluster_params.get('min_samples') is None else self.cluster_params.get('min_samples'),
                      metric='euclidean' if self.cluster_params.get('metric') is None else self.cluster_params.get('metric'),
                      metric_params=self.cluster_params.get('metric_params'),
                      algorithm='auto' if self.cluster_params.get('algorithm') is None else self.cluster_params.get('algorithm'),
                      leaf_size=30 if self.cluster_params.get('leaf_size') is None else self.cluster_params.get('leaf_size'),
                      p=self.cluster_params.get('p')
                      )

    def dbscan_param(self) -> dict:
        """
        Generate density-based algorithm for discovering clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(eps=np.random.uniform(low=0.1 if self.kwargs.get('eps_low') is None else self.kwargs.get('eps_low'),
                                          high=1.0 if self.kwargs.get('eps_high') is None else self.kwargs.get('eps_high')
                                          ),
                    min_samples=np.random.uniform(low=0.1 if self.kwargs.get('min_samples_low') is None else self.kwargs.get('min_samples_low'),
                                                  high=1.0 if self.kwargs.get('min_samples_high') is None else self.kwargs.get('min_samples_high')
                                                  ),
                    metric=np.random.choice(a=['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('algorithm') is None else self.kwargs.get('algorithm')),
                    leaf_size=np.random.randint(low=5 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=50 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                )
                    )

    def factor_analysis(self) -> FactorAnalysis:
        """
        Config factor analysis

        :return: FactorAnalysis
            Sklearn object containing the factor analysis configuration
        """
        return FactorAnalysis(n_components=None if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                              tol=0.01 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                              copy=True if self.cluster_params.get('copy') is None else self.cluster_params.get('copy'),
                              max_iter=1000 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                              noise_variance_init=None if self.cluster_params.get('noise_variance_init') is None else self.cluster_params.get('noise_variance_init'),
                              svd_method='randomized' if self.cluster_params.get('svd_method') is None else self.cluster_params.get('svd_method'),
                              iterated_power=3 if self.cluster_params.get('iterated_power') is None else self.cluster_params.get('iterated_power'),
                              random_state=self.seed
                              )

    def factor_analysis_param(self) -> dict:
        """
        Generate factor analysis parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=10 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    tol=np.random.uniform(low=0.001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.1 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=5000 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    svd_method=np.random.choice(a=['lapack', 'randomized'] if self.kwargs.get('svd_method') is None else self.kwargs.get('svd_method')),
                    iterated_power=np.random.randint(low=2 if self.kwargs.get('iterated_power_low') is None else self.kwargs.get('iterated_power_low'),
                                                     high=10 if self.kwargs.get('iterated_power_high') is None else self.kwargs.get('iterated_power_high')
                                                     ),
                    )

    def feature_agglomeration(self) -> FeatureAgglomeration:
        """
        Config feature agglomeration clustering

        :return: FeatureAgglomeration
            Sklearn object containing the feature agglomeration configuration
        """
        return FeatureAgglomeration(n_clusters=2 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                                    affinity='euclidean' if self.cluster_params.get('affinity') is None else self.cluster_params.get('affinity'),
                                    memory=None if self.cluster_params.get('memory') is None else self.cluster_params.get('memory'),
                                    connectivity=None if self.cluster_params.get('connectivity') is None else self.cluster_params.get('connectivity'),
                                    compute_full_tree='auto' if self.cluster_params.get('compute_full_tree') is None else self.cluster_params.get('compute_full_tree'),
                                    linkage='ward' if self.cluster_params.get('linkage') is None else self.cluster_params.get('linkage'),
                                    pooling_func=np.mean if self.cluster_params.get('pooling_func') is None else self.cluster_params.get('pooling_func'),
                                    distance_threshold=self.cluster_params.get('distance_threshold')
                                    )

    def feature_agglomeration_param(self) -> dict:
        """
        Generate feature agglomeration parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    metric=np.random.choice(a=['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    linkage=np.random.choice(a=['ward', 'complete', 'average', 'single'] if self.kwargs.get('linkage') is None else self.kwargs.get('linkage'))
                    )

    def independent_component_analysis(self) -> FastICA:
        """
        Config independent component analysis

        :return: FastICA
            Sklearn object containing the independent component analysis configuration
        """
        return FastICA(n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                       algorithm='parallel' if self.cluster_params.get('algorithm') is None else self.cluster_params.get('algorithm'),
                       whiten=True if self.cluster_params.get('whiten') is None else self.cluster_params.get('whiten'),
                       fun='logcosh' if self.cluster_params.get('fun') is None else self.cluster_params.get('fun'),
                       fun_args=None if self.cluster_params.get('fun_args') is None else self.cluster_params.get('fun_args'),
                       max_iter=200 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                       tol=0.0001 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                       w_init=None if self.cluster_params.get('w_init') is None else self.cluster_params.get('w_init'),
                       whiten_solver='svd' if self.cluster_params.get('whiten_solver') is None else self.cluster_params.get('whiten_solver'),
                       random_state=self.seed
                       )

    def independent_component_analysis_param(self) -> dict:
        """
        Generate independent component analysis parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    algorithm=np.random.choice(a=['parallel', 'deflation'] if self.kwargs.get('algorithm') is None else self.kwargs.get('algorithm')),
                    whiten=np.random.choice(a=[False, True] if self.kwargs.get('whiten') is None else self.kwargs.get('whiten')),
                    fun=np.random.choice(a=['logcosh', 'exp', 'cube'] if self.kwargs.get('fun') is None else self.kwargs.get('fun')),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    tol=np.random.uniform(low=0.00001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.001 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    whiten_solver=np.random.choice(a=['eigh', 'svd'] if self.kwargs.get('whiten_solver') is None else self.kwargs.get('whiten_solver'))
                    )

    def isometric_mapping(self) -> Isomap:
        """
        Config isometric mapping

        :return: Isomap
            Sklearn object containing the isometric mapping configuration
        """
        return Isomap(n_neighbors=5 if self.cluster_params.get('n_neighbors') is None else self.cluster_params.get('n_neighbors'),
                      n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                      eigen_solver='auto' if self.cluster_params.get('eigen_solver') is None else self.cluster_params.get('eigen_solver'),
                      tol=0 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                      max_iter=self.cluster_params.get('max_iter'),
                      path_method='auto' if self.cluster_params.get('path_method') is None else self.cluster_params.get('path_method'),
                      neighbors_algorithm='auto' if self.cluster_params.get('neighbors_algorithm') is None else self.cluster_params.get('neighbors_algorithm'),
                      metric='minkowski' if self.cluster_params.get('metric') is None else self.cluster_params.get('metric'),
                      p=2 if self.cluster_params.get('p') is None else self.cluster_params.get('p')
                      )

    def isometric_mapping_param(self) -> dict:
        """
        Generate isometric mapping analysis parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=20 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    eigen_solver=np.random.choice(a=['auto', 'arpack', 'dense'] if self.kwargs.get('eigen_solver') is None else self.kwargs.get('eigen_solver')),
                    tol=np.random.uniform(low=0.00001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.001 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    path_method=np.random.choice(a=['auto', 'FW', 'D'] if self.kwargs.get('path_method') is None else self.kwargs.get('path_method')),
                    neighbors_algorithm=np.random.choice(a=['eigh', 'svd'] if self.kwargs.get('neighbors_algorithm') is None else self.kwargs.get('neighbors_algorithm')),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p') is None else self.kwargs.get('p'))
                    )

    def kmeans(self):
        """
        Config k-means clustering

        :return KMeans
            Sklearn object containing the k-means clustering configuration
        """
        return KMeans(n_clusters=2 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                      init='random' if self.cluster_params.get('init') is None else self.cluster_params.get('init'),
                      n_init='auto' if self.cluster_params.get('n_init') is None else self.cluster_params.get('n_init'),
                      max_iter=300 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                      tol=1e-04 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                      algorithm='lloyd' if self.cluster_params.get('algorithm') is None else self.cluster_params.get('algorithm'),
                      random_state=self.seed
                      )

    def kmeans_param(self) -> dict:
        """
        Generate k-means parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    init=np.random.choice(a=['k-means++', 'random'] if self.kwargs.get('init') is None else self.kwargs.get('init')),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    tol=np.random.uniform(low=0.00001 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.001 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    algorithm=np.random.choice(a=['lloyd', 'elkan'] if self.kwargs.get('algorithm') is None else self.kwargs.get('algorithm'))
                    )

    def kmedians(self):
        """
        Config k-medians clustering

        :return kmedians
            pyclustering object containing the k-means clustering configuration
        """
        return kmedians(data=None if self.cluster_params.get('data') is None else self.cluster_params.get('data'),
                        initial_medians=None if self.cluster_params.get('initial_medians') is None else self.cluster_params.get('initial_medians'),
                        tolerance=0.001 if self.cluster_params.get('tolerance') is None else self.cluster_params.get('tolerance'),
                        ccore=True if self.cluster_params.get('ccore') is None else self.cluster_params.get('ccore')
                        )

    def kmedians_param(self) -> dict:
        """
        Generate k-medians parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    initializer=np.random.choice(a=['k-means++', 'random'] if self.kwargs.get('initializer') is None else self.kwargs.get('initializer')),
                    tolerance=np.random.uniform(low=0.0001 if self.kwargs.get('tolerance_low') is None else self.kwargs.get('tolerance_low'),
                                                high=0.01 if self.kwargs.get('tolerance_high') is None else self.kwargs.get('tolerance_high')
                                                )
                    )

    def kmedoids(self):
        """
        Config k-medoids clustering

        :return kmedoids
            pyclustering object containing the k-means clustering configuration
        """
        return kmedoids(data=None if self.cluster_params.get('data') is None else self.cluster_params.get('data'),
                        initial_index_medoids=None if self.cluster_params.get('initial_index_medoids') is None else self.cluster_params.get('initial_index_medoids'),
                        tolerance=0.0001 if self.cluster_params.get('tolerance') is None else self.cluster_params.get('tolerance'),
                        ccore=True if self.cluster_params.get('ccore') is None else self.cluster_params.get('ccore')
                        )

    def kmedoids_param(self) -> dict:
        """
        Generate k-medoids parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    initializer=np.random.choice(a=['k-means++', 'random'] if self.kwargs.get('initializer') is None else self.kwargs.get('initializer')),
                    tolerance=np.random.uniform(low=0.00001 if self.kwargs.get('tolerance_low') is None else self.kwargs.get('tolerance_low'),
                                                high=0.001 if self.kwargs.get('tolerance_high') is None else self.kwargs.get('tolerance_high')
                                                )
                    )

    def latent_dirichlet_allocation(self) -> LatentDirichletAllocation:
        """
        Config latent dirichlet allocation

        :return: LatentDirichletAllocation
            Sklearn object containing the latent dirichlet allocation configuration
        """
        return LatentDirichletAllocation(n_components=10 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                                         doc_topic_prior=None if self.cluster_params.get('doc_topic_prior') is None else self.cluster_params.get('doc_topic_prior'),
                                         topic_word_prior=None if self.cluster_params.get('topic_word_prior') is None else self.cluster_params.get('topic_word_prior'),
                                         learning_method='batch' if self.cluster_params.get('learning_method') is None else self.cluster_params.get('learning_method'),
                                         learning_decay=0.7 if self.cluster_params.get('learning_decay') is None else self.cluster_params.get('learning_decay'),
                                         learning_offset=10 if self.cluster_params.get('learning_offset') is None else self.cluster_params.get('learning_offset'),
                                         max_iter=10 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                                         batch_size=128 if self.cluster_params.get('batch_size') is None else self.cluster_params.get('batch_size'),
                                         evaluate_every=-1 if self.cluster_params.get('evaluate_every') is None else self.cluster_params.get('evaluate_every'),
                                         total_samples=1e6 if self.cluster_params.get('total_samples') is None else self.cluster_params.get('total_samples'),
                                         perp_tol=0.1 if self.cluster_params.get('perp_tol') is None else self.cluster_params.get('perp_tol'),
                                         mean_change_tol=0.001 if self.cluster_params.get('mean_change_tol') is None else self.cluster_params.get('mean_change_tol'),
                                         max_doc_update_iter=100 if self.cluster_params.get('max_doc_update_iter') is None else self.cluster_params.get('max_doc_update_iter'),
                                         verbose=0 if self.cluster_params.get('verbose') is None else self.cluster_params.get('verbose'),
                                         random_state=self.seed
                                         )

    def latent_dirichlet_allocation_param(self) -> dict:
        """
        Generate latent dirichlet allocation parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    learning_method=np.random.choice(a=['batch', 'online'] if self.kwargs.get('learning_method') is None else self.kwargs.get('learning_method')),
                    learning_decay=np.random.uniform(low=0.1 if self.kwargs.get('learning_decay_low') is None else self.kwargs.get('learning_decay_low'),
                                                     high=0.9 if self.kwargs.get('learning_decay_high') is None else self.kwargs.get('learning_decay_high')
                                                     ),
                    learning_offset=np.random.uniform(low=1.0 if self.kwargs.get('learning_offset_low') is None else self.kwargs.get('learning_offset_low'),
                                                      high=20.0 if self.kwargs.get('learning_offset_high') is None else self.kwargs.get('learning_offset_high')
                                                      ),
                    max_iter=np.random.randint(low=2 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=50 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    batch_size=np.random.choice(a=[8, 16, 32, 64, 128, 256, 512] if self.kwargs.get('batch_size') is None else self.kwargs.get('batch_size')),
                    total_samples=np.random.randint(low=1e3 if self.kwargs.get('total_samples_low') is None else self.kwargs.get('total_samples_low'),
                                                    high=1e8 if self.kwargs.get('total_samples_high') is None else self.kwargs.get('total_samples_high'),
                                                    ),
                    perp_tol=np.random.uniform(low=0.001 if self.kwargs.get('perp_tol_low') is None else self.kwargs.get('perp_tol_low'),
                                               high=0.5 if self.kwargs.get('perp_tol_high') is None else self.kwargs.get('perp_tol_high')
                                               ),
                    mean_change_tol=np.random.uniform(low=0.0001 if self.kwargs.get('mean_change_tol_low') is None else self.kwargs.get('mean_change_tol_low'),
                                                      high=0.1 if self.kwargs.get('mean_change_tol_high') is None else self.kwargs.get('mean_change_tol_high')
                                                      ),
                    max_doc_update_iter=np.random.randint(low=50 if self.kwargs.get('max_doc_update_iter_low') is None else self.kwargs.get('max_doc_update_iter_low'),
                                                          high=500 if self.kwargs.get('max_doc_update_iter_high') is None else self.kwargs.get('max_doc_update_iter_high')
                                                          )
                    )

    def locally_linear_embedding(self) -> LocallyLinearEmbedding:
        """
        Config locally linear embedding

        :return: LocallyLinearEmbedding
            Sklearn object containing the locally linear embedding configuration
        """
        return LocallyLinearEmbedding(n_neighbors=5 if self.cluster_params.get('n_neighbors') is None else self.cluster_params.get('n_neighbors'),
                                      n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                                      reg=0.001 if self.cluster_params.get('reg') is None else self.cluster_params.get('reg'),
                                      eigen_solver='auto' if self.cluster_params.get('eigen_solver') is None else self.cluster_params.get('eigen_solver'),
                                      tol=0.000001 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                                      max_iter=100 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                                      method='standard' if self.cluster_params.get('method') is None else self.cluster_params.get('method'),
                                      hessian_tol=0.0001 if self.cluster_params.get('hessian_tol') is None else self.cluster_params.get('hessian_tol'),
                                      modified_tol=1e-12 if self.cluster_params.get('modified_tol') is None else self.cluster_params.get('modified_tol'),
                                      neighbors_algorithm='auto' if self.cluster_params.get('neighbors_algorithm') is None else self.cluster_params.get('neighbors_algorithm'),
                                      random_state=self.seed
                                      )

    def locally_linear_embedding_param(self) -> dict:
        """
        Generate locally linear embedding parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_neighbors=np.random.randint(low=2 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=20 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    reg=np.random.uniform(low=0.0001 if self.kwargs.get('reg_low') is None else self.kwargs.get('reg_low'),
                                          high=0.1 if self.kwargs.get('reg_high') is None else self.kwargs.get('reg_high')
                                          ),
                    eigen_solver=np.random.choice(a=['auto', 'arpack', 'dense'] if self.kwargs.get('eigen_solver') is None else self.kwargs.get('eigen_solver')),
                    tol=np.random.uniform(low=1e-8 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=1e-3 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    method=np.random.choice(a=['standard', 'hessian', 'modified', 'ltsa'] if self.kwargs.get('method') is None else self.kwargs.get('method')),
                    hessian_tol=np.random.uniform(low=1e-8 if self.kwargs.get('hessian_tol_low') is None else self.kwargs.get('hessian_tol_low'),
                                                  high=1e-2 if self.kwargs.get('hessian_tol_high') is None else self.kwargs.get('hessian_tol_high')
                                                  ),
                    modified_tol=np.random.uniform(low=1e-16 if self.kwargs.get('modified_tol_low') is None else self.kwargs.get('modified_tol_low'),
                                                   high=1e-8 if self.kwargs.get('modified_tol_high') is None else self.kwargs.get('modified_tol_high')
                                                   ),
                    neighbors_algorithm=np.random.choice(a=['auto', 'brute', 'kd_tree', 'ball_tree'] if self.kwargs.get('neighbors_algorithm') is None else self.kwargs.get('neighbors_algorithm')),
                    )

    def multi_dimensional_scaling(self) -> MDS:
        """
        Config multi dimensional scaling

        :return: MDS
            Sklearn object containing the multi dimensional scaling configuration
        """
        return MDS(n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                   metric=True if self.cluster_params.get('metric') is None else self.cluster_params.get('metric'),
                   n_init=4 if self.cluster_params.get('n_init') is None else self.cluster_params.get('n_init'),
                   max_iter=300 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                   verbose=0 if self.cluster_params.get('verbose') is None else self.cluster_params.get('verbose'),
                   eps=0.001 if self.cluster_params.get('eps') is None else self.cluster_params.get('eps'),
                   dissimilarity='euclidean' if self.cluster_params.get('dissimilarity') is None else self.cluster_params.get('dissimilarity'),
                   random_state=self.seed,
                   )

    def multi_dimensional_scaling_param(self) -> dict:
        """
        Generate multi dimensional scaling parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    metric=np.random.choice(a=[False, True] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    n_init=np.random.randint(low=2 if self.kwargs.get('n_init_low') is None else self.kwargs.get('n_init_low'),
                                             high=10 if self.kwargs.get('n_init_high') is None else self.kwargs.get('n_init_high')
                                             ),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    eps=np.random.uniform(low=0.0001 if self.kwargs.get('eps_low') is None else self.kwargs.get('eps_low'),
                                          high=0.01 if self.kwargs.get('eps_high') is None else self.kwargs.get('eps_high')
                                          ),
                    dissimilarity=np.random.choice(a=['euclidean', 'precomputed'] if self.kwargs.get('dissimilarity') is None else self.kwargs.get('dissimilarity'))
                    )

    def non_negative_matrix_factorization(self) -> NMF:
        """
        Config non-negative matrix factorization

        :return NMF
            Sklearn object containing the non-negative matrix factorization clustering configuration
        """
        return NMF(n_components=10 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                   init=None if self.cluster_params.get('init') is None else self.cluster_params.get('init'),
                   solver='cd' if self.cluster_params.get('solver') is None else self.cluster_params.get('solver'),
                   beta_loss='frobenius' if self.cluster_params.get('beta_loss') is None else self.cluster_params.get('beta_loss'),
                   tol=0.0001 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                   max_iter=200 if self.cluster_params.get('max_iter') is None else self.cluster_params.get('max_iter'),
                   l1_ratio=0 if self.cluster_params.get('l1_ratio') is None else self.cluster_params.get('l1_ratio'),
                   verbose=0 if self.cluster_params.get('verbose') is None else self.cluster_params.get('verbose'),
                   shuffle=False if self.cluster_params.get('shuffle') is None else self.cluster_params.get('shuffle'),
                   random_state=self.seed,
                   )

    def non_negative_matrix_factorization_param(self) -> dict:
        """
        Generate non-negative matrix factorization parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    init=np.random.choice(a=['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'] if self.kwargs.get('init') is None else self.kwargs.get('init')),
                    solver=np.random.choice(a=['cd', 'mu'] if self.kwargs.get('solver') is None else self.kwargs.get('solver')),
                    beta_loss=np.random.choice(a=['frobenius', 'kullback-leibler', 'itakura-saito'] if self.kwargs.get('beta_loss') is None else self.kwargs.get('beta_loss')),
                    tol=np.random.uniform(low=1e-8 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=1e-2 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    max_iter=np.random.randint(low=50 if self.kwargs.get('max_iter_low') is None else self.kwargs.get('max_iter_low'),
                                               high=500 if self.kwargs.get('max_iter_high') is None else self.kwargs.get('max_iter_high')
                                               ),
                    l1_ratio=np.random.uniform(low=0 if self.kwargs.get('l1_ratio_low') is None else self.kwargs.get('l1_ratio_low'),
                                               high=1 if self.kwargs.get('l1_ratio_high') is None else self.kwargs.get('l1_ratio_high')
                                               ),
                    shuffle=np.random.choice(a=[False, True] if self.kwargs.get('shuffle') is None else self.kwargs.get('shuffle'))
                    )

    def optics(self) -> OPTICS:
        """
        Config ordering points to identify clustering structure

        :return: OPTICS
            Sklearn object containing the optics configuration
        """
        return OPTICS(min_samples=5 if self.cluster_params.get('min_samples') is None else self.cluster_params.get('min_samples'),
                      max_eps=np.inf if self.cluster_params.get('max_eps') is None else self.cluster_params.get('max_eps'),
                      metric='minkowski' if self.cluster_params.get('metric') is None else self.cluster_params.get('metric'),
                      p=2 if self.cluster_params.get('p') is None else self.cluster_params.get('p'),
                      metric_params=self.cluster_params.get('metric_params'),
                      cluster_method='xi' if self.cluster_params.get('cluster_method') is None else self.cluster_params.get('cluster_method'),
                      eps=self.cluster_params.get('eps'),
                      xi=0.05 if self.cluster_params.get('xi') is None else self.cluster_params.get('xi'),
                      predecessor_correction=True if self.cluster_params.get('predecessor_correction') is None else self.cluster_params.get('predecessor_correction'),
                      min_cluster_size=self.cluster_params.get('min_cluster_size'),
                      algorithm='auto' if self.cluster_params.get('algorithm') is None else self.cluster_params.get('algorithm'),
                      leaf_size=30 if self.cluster_params.get('leaf_size') is None else self.cluster_params.get('leaf_size')
                      )

    def optics_param(self) -> dict:
        """
        Generate ordering points to identify clustering structure parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(min_samples=np.random.randint(low=2 if self.kwargs.get('min_samples_low') is None else self.kwargs.get('min_samples_low'),
                                                  high=20 if self.kwargs.get('min_samples_high') is None else self.kwargs.get('min_samples_high')
                                                  ),
                    metric=np.random.choice(a=['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jacard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    p=np.random.choice(a=[1, 2, 3] if self.kwargs.get('p') is None else self.kwargs.get('p')),
                    cluster_method=np.random.choice(a=['xi', 'dbscan'] if self.kwargs.get('cluster_method') is None else self.kwargs.get('cluster_method')),
                    eps=np.random.uniform(low=0.1 if self.kwargs.get('eps_low') is None else self.kwargs.get('eps_low'),
                                          high=1.0 if self.kwargs.get('eps_high') is None else self.kwargs.get('eps_high')
                                          ),
                    xi=np.random.uniform(low=0.0 if self.kwargs.get('xi_low') is None else self.kwargs.get('xi_low'),
                                         high=1.0 if self.kwargs.get('xi_high') is None else self.kwargs.get('xi_high')
                                         ),
                    predecessor_correction=np.random.choice(a=[False, True] if self.kwargs.get('predecessor_correction') is None else self.kwargs.get('predecessor_correction')),
                    min_cluster_size=np.random.uniform(low=0.0 if self.kwargs.get('min_cluster_size_low') is None else self.kwargs.get('min_cluster_size_low'),
                                                       high=1.0 if self.kwargs.get('min_cluster_size_high') is None else self.kwargs.get('min_cluster_size_high')
                                                       ),
                    algorithm=np.random.choice(a=['auto', 'ball_tree', 'kd_tree', 'brute'] if self.kwargs.get('metric') is None else self.kwargs.get('metric')),
                    leaf_size=np.random.randint(low=10 if self.kwargs.get('leaf_size_low') is None else self.kwargs.get('leaf_size_low'),
                                                high=60 if self.kwargs.get('leaf_size_high') is None else self.kwargs.get('leaf_size_high')
                                                )
                    )

    def principal_component_analysis(self) -> PCA:
        """
        Config principal component analysis

        :return: PCA
            Sklearn object containing the principal component analysis configuration
        """
        return PCA(n_components=self.cluster_params.get('n_components'),
                   copy=True if self.cluster_params.get('copy') is None else self.cluster_params.get('copy'),
                   whiten=False if self.cluster_params.get('whiten') is None else self.cluster_params.get('whiten'),
                   svd_solver='auto' if self.cluster_params.get('svd_solver') is None else self.cluster_params.get('svd_solver'),
                   tol=0.0 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                   iterated_power='auto' if self.cluster_params.get('iterated_power') is None else self.cluster_params.get('iterated_power'),
                   n_oversamples=10 if self.cluster_params.get('n_oversamples') is None else self.cluster_params.get('n_oversamples'),
                   random_state=self.seed
                   )

    def principal_component_analysis_param(self) -> dict:
        """
        Generate principal component analysis parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    whiten=np.random.choice(a=[False, True] if self.kwargs.get('whiten') is None else self.kwargs.get('whiten')),
                    svd_solver=np.random.choice(a=['auto', 'full', 'arpack', 'randomized'] if self.kwargs.get('svd_solver') is None else self.kwargs.get('svd_solver')),
                    tol=np.random.uniform(low=0 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=np.inf if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          ),
                    iterated_power=np.random.uniform(low=0 if self.kwargs.get('iterated_power_low') is None else self.kwargs.get('iterated_power_low'),
                                                     high=np.inf if self.kwargs.get('iterated_power_high') is None else self.kwargs.get('iterated_power_high')
                                                     ),
                    n_oversamples=np.random.randint(low=5 if self.kwargs.get('n_oversamples_low') is None else self.kwargs.get('n_oversamples_low'),
                                                    high=20 if self.kwargs.get('n_oversamples_high') is None else self.kwargs.get('n_oversamples_high')
                                                    )
                    )

    def spectral_clustering(self) -> SpectralClustering:
        """
        Config spectral clustering

        :return: SpectralClustering
            Sklearn object containing the spectral clustering configuration
        """
        return SpectralClustering(n_clusters=8 if self.cluster_params.get('n_clusters') is None else self.cluster_params.get('n_clusters'),
                                  eigen_solver=self.cluster_params.get('eigen_solver'),
                                  n_init=10 if self.cluster_params.get('n_init') is None else self.cluster_params.get('n_init'),
                                  gamma=1.0 if self.cluster_params.get('gamma') is None else self.cluster_params.get('gamma'),
                                  affinity='rbf' if self.cluster_params.get('affinity') is None else self.cluster_params.get('affinity'),
                                  n_neighbors=10 if self.cluster_params.get('n_neighbors') is None else self.cluster_params.get('n_neighbors'),
                                  eigen_tol=0.0 if self.cluster_params.get('eigen_tol') is None else self.cluster_params.get('eigen_tol'),
                                  assign_labels='kmeans' if self.cluster_params.get('assign_labels') is None else self.cluster_params.get('assign_labels'),
                                  degree=3 if self.cluster_params.get('degree') is None else self.cluster_params.get('degree'),
                                  coef0=1 if self.cluster_params.get('coef0') is None else self.cluster_params.get('coef0'),
                                  kernel_params=self.cluster_params.get('kernel_params'),
                                  random_state=self.seed,
                                  )

    def spectral_clustering_param(self) -> dict:
        """
        Generate spectral clustering parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=4 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    eigen_solver=np.random.choice(a=['arpack', 'lobpcg', 'amg'] if self.kwargs.get('eigen_solver') is None else self.kwargs.get('eigen_solver')),
                    n_init=np.random.randint(low=2 if self.kwargs.get('n_init_low') is None else self.kwargs.get('n_init_low'),
                                             high=10 if self.kwargs.get('n_init_high') is None else self.kwargs.get('n_init_high')
                                             ),
                    gamma=np.random.uniform(low=0.1 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=2 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    affinity=np.random.choice(a=['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'] if self.kwargs.get('affinity') is None else self.kwargs.get('affinity')),
                    n_neighbors=np.random.randint(low=4 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=20 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  ),
                    assign_labels=np.random.choice(a=['kmeans', 'discretize', 'cluster_qr'] if self.kwargs.get('assign_labels') is None else self.kwargs.get('assign_labels')),
                    degree=np.random.randint(low=2 if self.kwargs.get('degree_low') is None else self.kwargs.get('degree_low'),
                                             high=4 if self.kwargs.get('degree_high') is None else self.kwargs.get('degree_high')
                                             ),
                    )

    def spectral_embedding(self) -> SpectralEmbedding:
        """
        Config spectral embedding

        :return: SpectralEmbedding
            Sklearn object containing the spectral embedding configuration
        """
        return SpectralEmbedding(n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                                 affinity='nearest_neighbors' if self.cluster_params.get('affinity') is None else self.cluster_params.get('affinity'),
                                 gamma=self.cluster_params.get('gamma'),
                                 eigen_solver=self.cluster_params.get('eigen_solver'),
                                 n_neighbors=self.cluster_params.get('n_neighbors'),
                                 random_state=self.seed
                                 )

    def spectral_embedding_param(self) -> dict:
        """
        Generate spectral embedding parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    affinity=np.random.choice(a=['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'] if self.kwargs.get('affinity') is None else self.kwargs.get('affinity')),
                    gamma=np.random.uniform(low=0.1 if self.kwargs.get('gamma_low') is None else self.kwargs.get('gamma_low'),
                                            high=2 if self.kwargs.get('gamma_high') is None else self.kwargs.get('gamma_high')
                                            ),
                    eigen_solver=np.random.choice(a=['arpack', 'lobpcg', 'amg'] if self.kwargs.get('eigen_solver') is None else self.kwargs.get('eigen_solver')),
                    n_neighbors=np.random.randint(low=4 if self.kwargs.get('n_neighbors_low') is None else self.kwargs.get('n_neighbors_low'),
                                                  high=20 if self.kwargs.get('n_neighbors_high') is None else self.kwargs.get('n_neighbors_high')
                                                  )
                    )

    def t_distributed_stochastic_neighbor_embedding(self) -> TSNE:
        """
        Config t-distributed stochastic neighbor embedding

        :return: TSNE
            Sklearn object containing the t-distributed stochastic neighbor embedding configuration
        """
        return TSNE(n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                    perplexity=30.0 if self.cluster_params.get('perplexity') is None else self.cluster_params.get('perplexity'),
                    early_exaggeration=12.0 if self.cluster_params.get('early_exaggeration') is None else self.cluster_params.get('early_exaggeration'),
                    learning_rate=200.0 if self.cluster_params.get('learning_rate') is None else self.cluster_params.get('learning_rate'),
                    n_iter=1000 if self.cluster_params.get('n_iter') is None else self.cluster_params.get('n_iter'),
                    n_iter_without_progress=300 if self.cluster_params.get('n_iter_without_progress') is None else self.cluster_params.get('n_iter_without_progress'),
                    min_grad_norm=1e-7 if self.cluster_params.get('min_grad_norm') is None else self.cluster_params.get('min_grad_norm'),
                    metric='euclidean' if self.cluster_params.get('metric') is None else self.cluster_params.get('metric'),
                    init='random' if self.cluster_params.get('init') is None else self.cluster_params.get('init'),
                    verbose=0 if self.cluster_params.get('verbose') is None else self.cluster_params.get('verbose'),
                    method='barnes_hut' if self.cluster_params.get('method') is None else self.cluster_params.get('method'),
                    angle=0.5 if self.cluster_params.get('angle') is None else self.cluster_params.get('angle'),
                    random_state=self.seed,
                    )

    def t_distributed_stochastic_neighbor_embedding_param(self) -> dict:
        """
        Generate t-distributed stochastic neighbor embedding parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    perplexity=np.random.randint(low=5 if self.kwargs.get('perplexity_low') is None else self.kwargs.get('perplexity_low'),
                                                 high=50 if self.kwargs.get('perplexity_high') is None else self.kwargs.get('perplexity_high')
                                                 ),
                    early_exaggeration=np.random.randint(low=5 if self.kwargs.get('early_exaggeration_low') is None else self.kwargs.get('early_exaggeration_low'),
                                                         high=30 if self.kwargs.get('early_exaggeration_high') is None else self.kwargs.get('early_exaggeration_high')
                                                         ),
                    learning_rate=np.random.uniform(low=10.0 if self.kwargs.get('learning_rate_low') is None else self.kwargs.get('learning_rate_low'),
                                                    high=1000.0 if self.kwargs.get('learning_rate_high') is None else self.kwargs.get('learning_rate_high')
                                                    ),
                    n_iter=np.random.randint(low=250 if self.kwargs.get('n_iter_low') is None else self.kwargs.get('n_iter_low'),
                                             high=2000 if self.kwargs.get('n_iter_high') is None else self.kwargs.get('n_iter_high')
                                             ),
                    n_iter_without_progress=np.random.randint(low=50 if self.kwargs.get('n_iter_without_progress_low') is None else self.kwargs.get('n_iter_without_progress_low'),
                                                              high=1000 if self.kwargs.get('n_iter_without_progress_high') is None else self.kwargs.get('n_iter_without_progress_high')
                                                              ),
                    min_grad_norm=np.random.uniform(low=1e-9 if self.kwargs.get('min_grad_norm_low') is None else self.kwargs.get('min_grad_norm_low'),
                                                    high=1e-3 if self.kwargs.get('min_grad_norm_high') is None else self.kwargs.get('min_grad_norm_high')
                                                    ),
                    init=np.random.choice(a=['random', 'pca'] if self.kwargs.get('init') is None else self.kwargs.get('init')),
                    method=np.random.choice(a=['barnes_hut', 'exact'] if self.kwargs.get('method') is None else self.kwargs.get('method')),
                    angle=np.random.uniform(low=0.2 if self.kwargs.get('angle_low') is None else self.kwargs.get('angle_low'),
                                            high=0.8 if self.kwargs.get('angle_high') is None else self.kwargs.get('angle_high')
                                            ),
                    )

    def truncated_single_value_decomp(self) -> TruncatedSVD:
        """
        Config latent semantic analysis using truncated single value decomposition

        :return: TruncatedSVD
            Sklearn object containing the latent truncated single value decomposition configuration
        """
        return TruncatedSVD(n_components=2 if self.cluster_params.get('n_components') is None else self.cluster_params.get('n_components'),
                            algorithm='randomized' if self.cluster_params.get('algorithm') is None else self.cluster_params.get('algorithm'),
                            n_iter=5 if self.cluster_params.get('n_iter') is None else self.cluster_params.get('n_iter'),
                            tol=0.0 if self.cluster_params.get('tol') is None else self.cluster_params.get('tol'),
                            random_state=self.seed
                            )

    def truncated_single_value_decomp_param(self) -> dict:
        """
        Generate latent semantic analysis using truncated single value decomposition parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_components=np.random.randint(low=2 if self.kwargs.get('n_components_low') is None else self.kwargs.get('n_components_low'),
                                                   high=20 if self.kwargs.get('n_components_high') is None else self.kwargs.get('n_components_high')
                                                   ),
                    algorithm=np.random.choice(a=['arpack', 'randomized'] if self.kwargs.get('algorithm') is None else self.kwargs.get('algorithm')),
                    n_iter=np.random.randint(low=2 if self.kwargs.get('n_iter_low') is None else self.kwargs.get('n_iter_low'),
                                             high=20 if self.kwargs.get('n_iter_high') is None else self.kwargs.get('n_iter_high')
                                             ),
                    tol=np.random.uniform(low=0 if self.kwargs.get('tol_low') is None else self.kwargs.get('tol_low'),
                                          high=0.2 if self.kwargs.get('tol_high') is None else self.kwargs.get('tol_high')
                                          )
                    )

    def xmeans(self):
        """
        Config x-means clustering

        :return xmeans
            pyclustering object containing the k-means clustering configuration
        """
        return xmeans(data=None if self.cluster_params.get('data') is None else self.cluster_params.get('data'),
                      initial_centers=None if self.cluster_params.get('initial_centers') is None else self.cluster_params.get('initial_centers'),
                      kmax=20 if self.cluster_params.get('initial_centers') is None else self.cluster_params.get('initial_centers'),
                      tolerance=0.001 if self.cluster_params.get('tolerance') is None else self.cluster_params.get('tolerance'),
                      criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION if self.cluster_params.get('criterion') is None else self.cluster_params.get('criterion'),
                      ccore=True if self.cluster_params.get('ccore') is None else self.cluster_params.get('ccore')
                      )

    def xmeans_param(self) -> dict:
        """
        Generate x-means parameter configuration randomly

        :return: dict
            Parameter config
        """
        return dict(n_clusters=np.random.randint(low=2 if self.kwargs.get('n_clusters_low') is None else self.kwargs.get('n_clusters_low'),
                                                 high=20 if self.kwargs.get('n_clusters_high') is None else self.kwargs.get('n_clusters_high')
                                                 ),
                    initializer=np.random.choice(a=['k-means++', 'random'] if self.kwargs.get('initializer') is None else self.kwargs.get('initializer')),
                    tolerance=np.random.uniform(low=0.00001 if self.kwargs.get('tolerance_low') is None else self.kwargs.get('tolerance_low'),
                                                high=0.001 if self.kwargs.get('tolerance_high') is None else self.kwargs.get('tolerance_high')
                                                ),
                    criterion=np.random.choice(a=[splitting_type.BAYESIAN_INFORMATION_CRITERION, splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH] if self.kwargs.get('criterion') is None else self.kwargs.get('criterion'))
                    )


class ModelGeneratorCluster(Clustering):
    """
    Class for generating (unsupervised) clustering models
    """
    def __init__(self,
                 model_name: str = None,
                 cluster_params: dict = None,
                 models: List[str] = None,
                 model_id: int = 0,
                 seed: int = 1234,
                 **kwargs
                 ):
        """
        :param model_name: str
            Abbreviated name of the model

        :param cluster_params: dict
            Pre-configured classification model parameter

        :param models: List[str]
            Names of the possible models to sample from

        :param model_id: int
            Model identifier

        :param seed: int
            Seed
        """
        super().__init__(cluster_params=cluster_params, seed=seed, **kwargs)
        self.id: int = model_id
        self.fitness: dict = {}
        self.fitness_score: float = 0.0
        self.models: List[str] = models
        self.model_name: str = model_name
        if self.model_name is None:
            self.random: bool = True
            if self.models is not None:
                for model in self.models:
                    if model not in CLUSTERING_ALGORITHMS.keys():
                        self.random: bool = False
                        raise UnsupervisedMLException(f'Model ({model}) is not supported. Supported classification models are: {list(CLUSTERING_ALGORITHMS.keys())}')
        else:
            if self.model_name not in CLUSTERING_ALGORITHMS.keys():
                raise UnsupervisedMLException(f'Model ({self.model_name}) is not supported. Supported classification models are: {list(CLUSTERING_ALGORITHMS.keys())}')
            else:
                self.random: bool = False
        self.model = None
        self.model_param: dict = {}
        self.model_param_mutated: dict = {}
        self.model_param_mutation: str = ''
        self.features: List[str] = []
        self.train_time = None
        self.creation_time: str = None
        self.cluster_type: str = None
        self.max_distance_of_partitioned_clusters: float = None

    def _distance_in_partitioning_clustering(self,
                                             x: np.ndarray,
                                             centroid: int,
                                             distance_metric: str = 'euclidean'
                                             ) -> float:
        """
        Calculate normalized distance metric for partitioning cluster algorithms

        :param x: np.ndaaray
            Data set

        :param centroid: int
            Centroid value

        :param distance_metric: str
            Name of the distance measurement to use
                -> euclidean: Euclidean pairwise distance

        :return: float
            Percentage of distance value divided by maximum distance value
        """
        _distance: float = pairwise_distances(X=x, Y=[centroid], metric=distance_metric)
        return (_distance / self.max_distance_of_partitioned_clusters) * 100

    def generate_model(self):
        """
        Generate (unsupervised) clustering model with randomized parameter configuration
        """
        if self.random:
            if self.models is None:
                self.model_name = copy.deepcopy(np.random.choice(a=CLUSTERING_ALGORITHMS.keys()))
            else:
                self.model_name = copy.deepcopy(np.random.choice(a=self.models))
            _model = copy.deepcopy(CLUSTERING_ALGORITHMS.get(self.model_name))
        else:
            _model = copy.deepcopy(CLUSTERING_ALGORITHMS.get(self.model_name))
        if len(self.cluster_params.keys()) == 0:
            self.model_param = getattr(Clustering(**self.kwargs), f'{_model}_param')()
            self.cluster_params = copy.deepcopy(self.model_param)
            _idx: int = 0 if len(self.model_param_mutated.keys()) == 0 else len(self.model_param_mutated.keys()) + 1
            self.model_param_mutated.update({str(_idx): {copy.deepcopy(self.model_name): {}}})
            for param in self.model_param.keys():
                self.model_param_mutated[str(_idx)][copy.deepcopy(self.model_name)].update({param: copy.deepcopy(self.model_param.get(param))})
        else:
            if len(self.model_param_mutation) > 0:
                self.model_param = getattr(Clustering(**self.kwargs), f'{_model}_param')()
                self.cluster_params = copy.deepcopy(self.model_param)
            else:
                self.model_param = copy.deepcopy(self.cluster_params)
        self.model_param_mutation = 'new_model'
        self.model = copy.deepcopy(getattr(Clustering(cluster_params=self.cluster_params), _model)())
        for cluster_type in CLUSTER_TYPES.keys():
            if _model in CLUSTER_TYPES[cluster_type]:
                self.cluster_type = cluster_type
                break
        if self.cluster_type is None:
            self.cluster_type = 'unknown'
        Log().log(msg=f'Generate clustering model: {self.model} (Cluster Type: {self.cluster_type})')

    def generate_params(self, param_rate: float = 0.1, force_param: dict = None):
        """
        Generate parameter for (unsupervised) clustering models

        :param param_rate: float
            Rate of parameters of each model to mutate

        :param force_param: dict
            Parameter config to force explicitly
        """
        if param_rate > 1:
            _rate: float = 1.0
        else:
            if param_rate > 0:
                _rate: float = param_rate
            else:
                _rate: float = 0.1
        _params: dict = getattr(Clustering(**self.kwargs), f'{CLUSTERING_ALGORITHMS.get(self.model_name)}_param')()
        _force_param: dict = {} if force_param is None else force_param
        _param_choices: List[str] = [p for p in _params.keys() if p not in _force_param.keys()]
        _gen_n_params: int = round(len(_params.keys()) * _rate)
        if _gen_n_params == 0:
            _gen_n_params = 1
        elif _gen_n_params > len(_param_choices):
            _gen_n_params = len(_param_choices)
        self.model_param_mutated.update({len(self.model_param_mutated.keys()) + 1: {copy.deepcopy(self.model_name): {}}})
        _new_model_params: dict = copy.deepcopy(self.model_param)
        _already_mutated_params: List[str] = []
        for param in _force_param.keys():
            _new_model_params.update({param: _force_param.get(param)})
        for _ in range(0, _gen_n_params, 1):
            _param: str = np.random.choice(a=_param_choices)
            if _param in _already_mutated_params:
                _counter: int = 0
                while _param not in _already_mutated_params or _counter > 100:
                    _param: str = np.random.choice(a=_param_choices)
                    _counter += 1
            _already_mutated_params.append(_param)
            if _params.get(_param) == _new_model_params.get(_param):
                _counter: int = 0
                _next_attempt: dict = getattr(Clustering(**self.kwargs), f'{CLUSTERING_ALGORITHMS.get(self.model_name)}_param')()
                while _next_attempt.get(_param) == _new_model_params.get(_param) or _counter > 100:
                    _next_attempt: dict = getattr(Clustering(**self.kwargs), f'{CLUSTERING_ALGORITHMS.get(self.model_name)}_param')()
                    _counter += 1
                _new_model_params.update({_param: copy.deepcopy(_next_attempt.get(_param))})
            else:
                _new_model_params.update({_param: copy.deepcopy(_params.get(_param))})
            Log().log(msg=f'Change hyperparameter: {_param} from {self.model_param.get(_param)} to {_new_model_params.get(_param)} of model {self.model_name}')
            self.model_param_mutated[list(self.model_param_mutated.keys())[-1]][copy.deepcopy(self.model_name)].update({_param: _params.get(_param)})
        self.model_param_mutation = 'params'
        self.model_param = copy.deepcopy(_new_model_params)
        self.cluster_params = self.model_param
        self.model = getattr(Clustering(clf_params=self.cluster_params, **self.kwargs), self.model_name)()
        Log().log(msg=f'Generate hyperparameter for clustering model: Rate={param_rate}, Changed={self.model_param_mutated}')

    def get_model_parameter(self) -> dict:
        """
        Get parameter "standard" config of given regression models

        :return dict:
            Standard parameter config of given regression models
        """
        _model_param: dict = {}
        if self.models is None:
            return _model_param
        else:
            for model in self.models:
                if model in CLUSTERING_ALGORITHMS.keys():
                    _model = getattr(Clustering(), model)()
                    _param: dict = getattr(Clustering(), f'{model}_param')()
                    _model_random_param: dict = _model.__dict__.items()
                    for param in _model_random_param:
                        if param[0] in _param.keys():
                            _param.update({param[0]: param[1]})
                    _model_param.update({model: copy.deepcopy(_param)})
                    Log().log(msg=f'Get standard hyperparameter for clustering model: {model}')
        return _model_param

    def predict(self, x: np.ndarray) -> Tuple[np.array, float]:
        """
        Get prediction from trained (unsupervised) clustering model

        :param x: np.ndarray
            Test data set

        :return Tuple[np.array, float]:
            Predicted cluster classes, normalized distance metric
        """
        if hasattr(self.model, 'predict'):
            _pred: np.ndarray = self.model.predict(x).flatten()
            return _pred, self._distance_in_partitioning_clustering(x=x,
                                                                    centroid=1,
                                                                    distance_metric='euclidean'
                                                                    )
        else:
            raise UnsupervisedMLException(f'Model ({self.model_name}) has no function called "predict"')

    def train(self, x: np.ndarray):
        """
        Train or fit supervised machine learning model

        :param x: np.ndarray
            Train data set
        """
        Log().log(msg=f'Train classifier: Model={self.model_name}, Cases={x.shape[0]}, Predictors={x.shape[1]}, Hyperparameter={self.model_param}')
        _t0: datetime = datetime.now()
        if hasattr(self.model, 'fit'):
            if hasattr(self.model, 'fit_transform'):
                self.model.fit_transform(x)
            else:
                self.model.fit(x)
        elif hasattr(self.model, 'process'):
            if self.cluster_params['initializer'] == 'kmeans++':
                _initial_cluster_center: np.ndarray = kmeans_plusplus_initializer(data=x,
                                                                                  amount_centers=self.model_param['n_clusters'],
                                                                                  amount_candidates=None
                                                                                  ).initialize()
            elif self.cluster_params['initializer'] == 'random':
                _initial_cluster_center: np.ndarray = random_center_initializer(data=x,
                                                                                amount_centers=self.model_param['n_clusters'],
                                                                                ).initialize()
            else:
                raise UnsupervisedMLException(f'Cluster intitializer ({self.cluster_params["initializer"]}) not supported')
            self.cluster_params = copy.deepcopy(self.model_param)
            self.cluster_params.update({SPECIAL_PARAMS.get(self.model_name): _initial_cluster_center})
            self.model = copy.deepcopy(getattr(Clustering(cluster_params=self.cluster_params), self.model_name)())
            self.model.process()
            self.cluster_params = None
        else:
            raise UnsupervisedMLException('Training (fitting) method not supported by given model object')
        self.train_time = (datetime.now() - _t0).seconds
        self.creation_time = str(datetime.now())
        if self.cluster_type == 'partition':
            self.max_distance_of_partitioned_clusters = pairwise_distances(X=x, metric='euclidean').max()
        Log().log(msg=f'Clustering model trained after {self.train_time} seconds')


class UnsupervisedMLException(Exception):
    """
    Class for handling exceptions for classes ModelGeneratorCluster, ClusterVisualization
    """
    pass


class ClusterVisualization:
    """
    Class for preparing data for cluster visualization
    """
    def __init__(self,
                 model_generator: ModelGeneratorCluster,
                 df: pd.DataFrame,
                 features: List[str] = None,
                 find_optimum: bool = True,
                 silhouette_analysis: bool = True,
                 n_cluster_components: int = None,
                 n_neighbors: int = None,
                 n_iter: int = None,
                 metric: List[str] = None,
                 affinity: List[str] = None,
                 connectivity: List[str] = None,
                 linkage: List[str] = None,
                 **kwargs
                 ):
        """
        :param df: pd.DataFrame
            Data set

        :param features: List[str]
            List of strings containing the features to cluster

        :param target: str
            Name of the target features

        :param find_optimum: bool
            Whether to Find optimum number of components or clusters or not

        :param n_cluster_components: int
            Amount of clusters for partitioning clustering

        :param n_neighbors: int
            Amount of neighbors

        :param n_iter: int
            Amount of iterations

        :param metric: List[str]
            Names of the metric for each clustering

        :param affinity: List[str]
            Names of the affinity metric for each clustering

        :param connectivity: List[str]
            Names of the connectivity structure for each clustering

        :param linkage: List[str]
            Names of the linkage function for each clustering

        :param silhouette_analysis: bool
            Run silhouette analysis to evaluate clustering or not

        :param kwargs: dict
            Key-word arguments regarding the machine learning algorithms
        """
        self.model_generator: ModelGeneratorCluster = model_generator
        self.df: pd.DataFrame = df
        self.features: List[str] = list(self.df.keys()) if features is None else features
        if len(self.features) == 0:
            self.features = list(self.df.keys())
        self.cluster: dict = {}
        self.cluster_plot: dict = {}
        self.ml_algorithm: str = None
        self.find_optimum: bool = find_optimum
        self.n_cluster_components: int = n_cluster_components
        self.n_neighbors: int = n_neighbors
        self.n_iter: int = n_iter
        self.metric: List[str] = metric
        self.affinity: List[str] = affinity
        self.connectivity: List[str] = connectivity
        self.linkage: List[str] = linkage
        self.silhouette: bool = silhouette_analysis
        self.seed: int = 1234
        self.eigen_value = None
        self.eigen_vector = None
        self.kwargs: dict = {} if kwargs is None else kwargs
        if self.kwargs.get('file_path') is None:
            self.to_export: bool = False
        else:
            if len(self.kwargs.get('file_path')) > 0:
                self.to_export: bool = True
            else:
                self.to_export: bool = False

    def _affinity_propagation(self) -> None:
        """
        Affinity propagation for graph based cluster without pre-defined partitions
        """
        _labels: np.array = self.model_generator.model.predict(self.df[self.features])
        self.cluster[self.ml_algorithm].update({'n_clusters': len(list(set(_labels)))})
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'affinity_propagation_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'affinity_propagation_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_labels)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Affinity Propagation: Silhouette Analysis': dict(data=self.df,
                                                                                            features=None,
                                                                                            plot_type='silhouette',
                                                                                            use_auto_extensions=False if self.kwargs.get(
                                                                                                'use_auto_extensions') is None else self.kwargs.get(
                                                                                                'use_auto_extensions'),
                                                                                            file_path=_file_path_silhouette,
                                                                                            kwargs=dict(layout={},
                                                                                                        n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                        silhouette=_silhouette
                                                                                                        )
                                                                                            )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'cluster_centers': self.model_generator.model.cluster_centers_,
                                                'affinity_matrix': self.model_generator.model.affinity_matrix_,
                                                'labels': self.model_generator.model.labels_,
                                                'cluster': self.model_generator.model.predict(X=self.df[self.features])
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'Affinity Propagation: Cluster Partition': dict(data=_df,
                                                                                  features=self.features,
                                                                                  group_by=['cluster'],
                                                                                  melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                                  plot_type='scatter',
                                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                                      'use_auto_extensions') is None else self.kwargs.get(
                                                                                      'use_auto_extensions'),
                                                                                  file_path=_file_path_cluster_partition,
                                                                                  kwargs=dict(layout={})
                                                                                  )
                                  })

    def _agglomerative_clustering(self) -> None:
        """
        Agglomerative clustering for hierarchical clustering using similarities
        """
        if self.ml_algorithm.find('unstruc') < 0:
            if self.kwargs.get('connectivity') is None:
                self.kwargs.update({'connectivity': kneighbors_graph(X=self.df[self.features],
                                                                     n_neighbors=self.n_neighbors,
                                                                     mode='connectivity' if self.kwargs.get(
                                                                         'connectivity') is None else self.kwargs.get(
                                                                         'connectivity'),
                                                                     metric='minkowski' if self.kwargs.get(
                                                                         'metric') is None else self.kwargs.get(
                                                                         'metric'),
                                                                     p=2 if self.kwargs.get(
                                                                         'p') is None else self.kwargs.get('p'),
                                                                     metric_params=self.kwargs.get('metric_params'),
                                                                     include_self=False if self.kwargs.get(
                                                                         'include_self') is None else self.kwargs.get(
                                                                         'include_self'),
                                                                     n_jobs=os.cpu_count()
                                                                     )
                                    })
        #_clustering: AgglomerativeClustering = Clustering(cluster_params=self.kwargs).agglomerative_clustering()
        self.cluster[self.ml_algorithm].update({'n_clusters': self.model_generator.model.n_clusters_})
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_partition.html')
            _file_path_hierarchical: str = os.path.join(self.kwargs.get('file_path'), 'agglomerative_clustering_hierarchical.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
            _file_path_hierarchical: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Agglomerative Clustering: Silhouette Analysis': dict(data=self.df,
                                                                                                features=None,
                                                                                                plot_type='silhouette',
                                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                                    'use_auto_extensions') is None else self.kwargs.get(
                                                                                                    'use_auto_extensions'),
                                                                                                file_path=_file_path_silhouette,
                                                                                                kwargs=dict(layout={},
                                                                                                            n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                            silhouette=_silhouette
                                                                                                            )
                                                                                                )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'connectivity': self.kwargs.get('connectivity'),
                                                'n_clusters': self.model_generator.model.n_clusters_,
                                                'n_leaves': self.model_generator.model.n_leaves_,
                                                'n_components': self.model_generator.model.n_connected_components_,
                                                'children': self.model_generator.model.children_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Agglomerative Clustering: Partition': dict(data=_df,
                                                                              features=self.features,
                                                                              group_by=['cluster'],
                                                                              melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                              plot_type='scatter',
                                                                              use_auto_extensions=False if self.kwargs.get('use_auto_extensions') is None else self.kwargs.get('use_auto_extensions'),
                                                                              file_path=_file_path_cluster_partition,
                                                                              kwargs=dict(layout={})
                                                                              ),
                                  'Agglomerative Clustering: Hierarchical': dict(data=_df,
                                                                                 features=self.features,
                                                                                 plot_type='dendro',
                                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                                     'use_auto_extensions') is None else self.kwargs.get(
                                                                                     'use_auto_extensions'),
                                                                                 file_path=_file_path_hierarchical,
                                                                                 kwargs=dict(layout={})
                                                                                 ),
                                  })

    def _birch_clustering(self) -> None:
        """
        Balanced iterative reducing and clustering using hierarchies (Birch) for generating efficient cluster partitions on big data sets
        """
        _labels: np.array = self.model_generator.model.predict(self.df[self.features])
        self.cluster[self.ml_algorithm].update({'n_clusters': len(list(set(_labels)))})
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'birch_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'birch_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=_labels)
                self.cluster[self.ml_algorithm].update({'n_clusters': int((len(list(_silhouette.keys())) - 1) / 2),
                                                        'silhouette': _silhouette
                                                        })
                self.cluster_plot.update({'Birch: Silhouette Analysis': dict(data=self.df,
                                                                             features=None,
                                                                             plot_type='silhouette',
                                                                             use_auto_extensions=False if self.kwargs.get(
                                                                                 'use_auto_extensions') is None else self.kwargs.get(
                                                                                 'use_auto_extensions'),
                                                                             file_path=_file_path_silhouette,
                                                                             kwargs=dict(layout={},
                                                                                         n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                         silhouette=_silhouette
                                                                                         )
                                                                             )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'partial_fit': self.model_generator.model.partial_fit_,
                                                'root': self.model_generator.model.root_,
                                                'centroids': self.model_generator.model.subcluster_centers_,
                                                'cluster': self.model_generator.model.transform(X=self.df[self.features]),
                                                'cluster_labels': self.model_generator.model.subcluster_labels_,
                                                'dummy_leaf': self.model_generator.model.dummy_leaf_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Birch: Cluster Partition': dict(data=_df,
                                                                   features=self.features,
                                                                   group_by=['cluster'],
                                                                   melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                   plot_type='scatter',
                                                                   use_auto_extensions=False if self.kwargs.get(
                                                                       'use_auto_extensions') is None else self.kwargs.get(
                                                                       'use_auto_extensions'),
                                                                   file_path=_file_path_cluster_partition,
                                                                   kwargs=dict(layout={})
                                                                   )
                                  })

    def _clean_missing_data(self) -> None:
        """
        Clean cases containing missing data
        """
        Log(write=False, level='info').log(msg='Clean cases containing missing values...')
        self.df.dropna(axis=0, how='any', inplace=True)
        if self.df.shape[0] == 0:
            raise UnsupervisedMLException('No cases containing valid observations left')

    def _cumulative_explained_variance_ratio(self, explained_variance_ratio: np.ndarray) -> int:
        """
        Calculate optimal amount of components to be used for principal component analysis based on the explained variance ratio

        :return: int
            Optimal amount of components
        """
        _threshold: float = 0.75 if self.kwargs.get('cev') is None else self.kwargs.get('cev')
        for i, ratio in enumerate(np.cumsum(explained_variance_ratio)):
            if ratio >= _threshold:
                return i + 1

    def _density_based_spatial_clustering_applications_with_noise(self) -> None:
        """
        Density-based spatial clustering applications with noise (DBSCAN) for clustering complex structures like dense regions in space
        """
        self.cluster[self.ml_algorithm].update({'n_clusters': len(list(set(self.model_generator.model.labels_))),
                                                'core_sample_indices': self.model_generator.model.core_sample_indices_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        if self.to_export:
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'dbscan_cluster_partition.html')
        else:
            _file_path_cluster_partition: str = None
        self.cluster_plot.update({'DBSCAN: Cluster Partition': dict(data=_df,
                                                                    features=self.features,
                                                                    group_by=['cluster'],
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                        'use_auto_extensions') is None else self.kwargs.get(
                                                                        'use_auto_extensions'),
                                                                    file_path=_file_path_cluster_partition,
                                                                    kwargs=dict(layout={})
                                                                    )
                                  })

    def _factor_analysis(self) -> None:
        """
        Factor analysis
        """
        _kmo: dict = self._factoriability_test(meth='kmo')
        self.cluster[self.ml_algorithm].update({'kmo': _kmo})
        if _kmo.get('kmo') < 0.6:
            Log().log(msg=f'Data set not suitable for running factor analysis since KMO coefficient ({_kmo.get("kmo")}) is lower than 0.6')
        else:
            if self.n_cluster_components is None:
                self.kwargs.update({'n_factors': 2})
            else:
                if self.n_cluster_components >= len(self.features):
                    self.kwargs.update({'n_factors': 2})
                    Log().log(msg='Number of factors are greater than or equal to number of features. Number of factors set to 2')
                else:
                    self.kwargs.update({'n_components': self.n_cluster_components})
            self.cluster[self.ml_algorithm].update({'n_factors': self.kwargs.get('n_factors')})
            if self.find_optimum:
                if self.silhouette:
                    _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.transform(self.df[self.features]))
                    self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                    self.cluster_plot.update({'Silhouette Analysis (FA)': dict(data=None,
                                                                               features=None,
                                                                               plot_type='silhouette',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                   'use_auto_extensions') is None else self.kwargs.get(
                                                                                   'use_auto_extensions'),
                                                                               file_path=self.kwargs.get('file_path'),
                                                                               kwargs=dict(layout={},
                                                                                           n_clusters=self.kwargs.get(
                                                                                               'n_clusters'),
                                                                                           silhouette=_silhouette
                                                                                           )
                                                                               )
                                              })
                else:
                    self.kwargs.update({'n_factors': self._estimate_optimal_factors(factors=self.model_generator.model.transform(X=self.df[self.features]))})
                    self.cluster[self.ml_algorithm].update({'n_factors': self.kwargs.get('n_factors')})
                    self.cluster_plot.update({'Optimal Number of Factors': dict(data=self.eigen_value,
                                                                                features=None,
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                    'use_auto_extensions') is None else self.kwargs.get(
                                                                                    'use_auto_extensions'),
                                                                                file_path=self.kwargs.get('file_path'),
                                                                                kwargs=dict(layout={})
                                                                                )
                                              })
            if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
                self.cluster[self.ml_algorithm].update({'silhouette': None})
            _factors: np.array = self.model_generator.model.transform(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'factors': self.model_generator.model.components_,
                                                    'explained_variance': self.model_generator.model.explained_variance_,
                                                    'fa': _factors
                                                    })
            _components: pd.DataFrame = pd.DataFrame(data=np.array(self.model_generator.model.components_),
                                                     columns=self.features,
                                                     index=['fa{}'.format(fa) for fa in
                                                            range(0, self.kwargs.get('n_factors'), 1)]
                                                     ).transpose()
            _feature_importance: pd.DataFrame = abs(_components)
            for fa in range(0, self.kwargs.get('n_factors'), 1):
                self.cluster_plot.update({'Feature Importance FA{}'.format(fa): dict(data=_feature_importance,
                                                                                     features=None,
                                                                                     plot_type='bar',
                                                                                     use_auto_extensions=False if self.kwargs.get(
                                                                                         'use_auto_extensions') is None else self.kwargs.get(
                                                                                         'use_auto_extensions'),
                                                                                     file_path=self.kwargs.get(
                                                                                         'file_path'),
                                                                                     kwargs=dict(layout={},
                                                                                                 x=self.features,
                                                                                                 y=_feature_importance[
                                                                                                     'fa{}'.format(fa)],
                                                                                                 marker=dict(color=
                                                                                                             _feature_importance[
                                                                                                                 'fa{}'.format(
                                                                                                                     fa)],
                                                                                                             colorscale='rdylgn',
                                                                                                             autocolorscale=True
                                                                                                             )
                                                                                                 )
                                                                                     )
                                          })
            self.cluster_plot.update({'Explained Variance': dict(data=pd.DataFrame(),
                                                                 features=None,
                                                                 plot_type='bar',
                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                     'use_auto_extensions') is None else self.kwargs.get(
                                                                     'use_auto_extensions'),
                                                                 file_path=self.kwargs.get('file_path'),
                                                                 kwargs=dict(layout={},
                                                                             x=list(_feature_importance.keys()),
                                                                             y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                             )
                                                                 ),
                                      'Factor Loadings': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('fa'),
                                                                                columns=list(_feature_importance.keys())
                                                                                ),
                                                              features=list(_feature_importance.keys()),
                                                              plot_type='scatter',
                                                              use_auto_extensions=False if self.kwargs.get(
                                                                  'use_auto_extensions') is None else self.kwargs.get(
                                                                  'use_auto_extensions'),
                                                              file_path=self.kwargs.get('file_path'),
                                                              melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                              kwargs=dict(layout={},
                                                                          marker=dict(color=self.cluster[self.ml_algorithm].get('fa'),
                                                                                      colorscale='rdylgn',
                                                                                      autocolorscale=True
                                                                                      )
                                                                          )
                                                              )
                                      })

    def _factoriability_test(self, meth: str = 'kmo') -> dict:
        """
        Test whether a data set contains unobserved features required for factor analysis

        :param meth: str
            Name of the used method
                -> kmo: Kaiser-Meyer-Olkin Criterion
                -> bartlette: Bartlette's test of sphericity
        """
        _fac: dict = {}
        if meth == 'kmo':
            pass
        elif meth == 'bartlette':
            pass
        else:
            raise UnsupervisedMLException(f'Method for testing "factoriability" ({meth}) not supported')
        return _fac

    def _feature_agglomeration(self) -> None:
        """
        Feature agglomeration for reducing features into grouped clusters (hierarchical clustering)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log(write=False,
                    level='info'
                    ).log(msg=f"It makes no sense to reduce feature dimensionality into less than 2 groups ({self.kwargs.get('n_clusters')}). Number of clusters are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_clusters': self.model_generator.model.n_clusters_,
                                                'n_leaves': self.model_generator.model.n_leaves_,
                                                'n_components': self.model_generator.model.n_connected_components_,
                                                'children': self.model_generator.model.children_,
                                                'reduced_data_set': self.model_generator.model.transform(X=self.df[self.features]),
                                                'labels': self.model_generator.model.labels_
                                                })

    def _estimate_optimal_factors(self, factors: np.array) -> int:
        """
        Calculate optimal amount of factors to be used in factor analysis based on the eigenvalues

        :param factors: np.array
            Factor loadings

        :return: int
            Optimal amount of factors
        """
        _diff: List[float] = []
        self.eigen_value, self.eigen_vector = np.linalg.eig(factors)
        for eigen_value in self.eigen_value:
            _diff.append(1 - eigen_value)
        return _diff.index(factors.min())

    def _elbow(self) -> int:
        """
        Calculate optimal number of clusters for partitioning clustering

        :return: int
            Optimal amount of clusters
        """
        _distortions: list = []
        _max_clusters: int = 10 if self.kwargs.get('max_clusters') is None else self.kwargs.get('max_clusters')
        for i in range(1, _max_clusters, 1):
            self.kwargs.update({'n_clusters': i})
            _distortions.append(self.model_generator.model.inertia_)
        return 1

    def _isometric_mapping(self) -> None:
        """
        Isometric mapping for non-linear dimensionality reduction (manifold learning)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 3})
        else:
            self.kwargs.update({'n_components': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'embeddings': self.model_generator.model.embedding_,
                                                'transformed_embeddings': self.model_generator.model.transform(X=self.df[self.features]),
                                                'distance_matrix': self.model_generator.model.dist_matrix_,
                                                'kernel_pca': self.model_generator.model.kernel_pca_,
                                                'reconstruction_error': self.model_generator.model.reconstruction_error()
                                                })

    def _k_means(self) -> None:
        """
        K-Means clustering (partitioning clustering) for graphical data, which is spherical about the cluster centre
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log().log(msg=f"It makes no sense to run cluster analysis with less than 2 clusters ({self.kwargs.get('n_clusters')}). Number of components are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_clusters': self.kwargs.get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'kmeans_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'kmeans_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.predict(self.df[self.features]))
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'K-Means: Silhouette Analysis': dict(data=self.df,
                                                                               features=None,
                                                                               plot_type='silhouette',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                   'use_auto_extensions') is None else self.kwargs.get(
                                                                                   'use_auto_extensions'),
                                                                               file_path=_file_path_silhouette,
                                                                               kwargs=dict(layout={},
                                                                                           n_clusters=self.kwargs.get(
                                                                                               'n_clusters'),
                                                                                           silhouette=_silhouette
                                                                                           )
                                                                               )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'inertia': self.model_generator.model.inertia_,
                                                'cluster': self.model_generator.model.predict(X=self.df[self.features]),
                                                'cluster_distance_space': self.model_generator.model.transform(X=self.df[self.features]),
                                                'centroids': self.model_generator.model.cluster_centers_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'K-Means: Cluster Partition': dict(data=_df,
                                                                     features=self.features,
                                                                     group_by=['cluster'],
                                                                     melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                     plot_type='scatter',
                                                                     use_auto_extensions=False if self.kwargs.get(
                                                                         'use_auto_extensions') is None else self.kwargs.get(
                                                                         'use_auto_extensions'),
                                                                     file_path=_file_path_cluster_partition,
                                                                     kwargs=dict(layout={})
                                                                     )
                                  })

    def _k_medians(self) -> None:
        """
        K-Medians clustering (partitioning clustering) for graphical data, which is spherical about the cluster centre
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log().log(msg=f"It makes no sense to run cluster analysis with less than 2 clusters ({self.kwargs.get('n_clusters')}). Number of components are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_clusters': self.kwargs.get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'kmedians_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'kmedians_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.predict(self.df[self.features]))
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'K-Medians: Silhouette Analysis': dict(data=self.df,
                                                                                 features=None,
                                                                                 plot_type='silhouette',
                                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                                     'use_auto_extensions') is None else self.kwargs.get(
                                                                                     'use_auto_extensions'),
                                                                                 file_path=_file_path_silhouette,
                                                                                 kwargs=dict(layout={},
                                                                                             n_clusters=self.kwargs.get(
                                                                                                 'n_clusters'),
                                                                                             silhouette=_silhouette
                                                                                             )
                                                                                 )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'cluster': self.model_generator.model.predict(X=self.df[self.features]),
                                                'medians': self.model_generator.model.get_medians(),
                                                'metric': self.model_generator.model.get_total_wce()
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'K-Medians: Cluster Partition': dict(data=_df,
                                                                       features=self.features,
                                                                       group_by=['cluster'],
                                                                       melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                       plot_type='scatter',
                                                                       use_auto_extensions=False if self.kwargs.get(
                                                                           'use_auto_extensions') is None else self.kwargs.get(
                                                                           'use_auto_extensions'),
                                                                       file_path=_file_path_cluster_partition,
                                                                       kwargs=dict(layout={})
                                                                       )
                                  })

    def _k_medoids(self) -> None:
        """
        K-Medoids clustering (partitioning clustering) for graphical data, which is spherical about the cluster centre
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log().log(msg=f"It makes no sense to run cluster analysis with less than 2 clusters ({self.kwargs.get('n_clusters')}). Number of components are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_clusters': self.kwargs.get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'kmedoids_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'kmedoids_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.predict(self.df[self.features]))
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'K-Medoids: Silhouette Analysis': dict(data=self.df,
                                                                                 features=None,
                                                                                 plot_type='silhouette',
                                                                                 use_auto_extensions=False if self.kwargs.get(
                                                                                     'use_auto_extensions') is None else self.kwargs.get(
                                                                                     'use_auto_extensions'),
                                                                                 file_path=_file_path_silhouette,
                                                                                 kwargs=dict(layout={},
                                                                                             n_clusters=self.kwargs.get(
                                                                                                 'n_clusters'),
                                                                                             silhouette=_silhouette
                                                                                             )
                                                                                 )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'cluster': self.model_generator.model.predict(X=self.df[self.features]),
                                                'medoids': self.model_generator.model.get_medoids()
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'K-Medoids: Cluster Partition': dict(data=_df,
                                                                       features=self.features,
                                                                       group_by=['cluster'],
                                                                       melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                       plot_type='scatter',
                                                                       use_auto_extensions=False if self.kwargs.get(
                                                                           'use_auto_extensions') is None else self.kwargs.get(
                                                                           'use_auto_extensions'),
                                                                       file_path=_file_path_cluster_partition,
                                                                       kwargs=dict(layout={})
                                                                       )
                                  })

    def _latent_dirichlet_allocation(self) -> None:
        """
        Latent Dirichlet Allocation for text clustering
        """
        self.cluster[self.ml_algorithm].update({'components': self.model_generator.model.transform(X=self.df[self.features]),
                                                'em_iter': self.model_generator.model.n_batch_iter_,
                                                'passes_iter': self.model_generator.model.n_iter_,
                                                'perplexity_score': self.model_generator.model.bound_,
                                                'doc_topic_prior': self.model_generator.model.doc_topic_prior_,
                                                'topic_word_prior': self.model_generator.model.topic_word_prior_,
                                                })

    def _locally_linear_embedding(self) -> None:
        """
        Locally linear embedding for non-linear dimensionality reduction (manifold learning)
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'embeddings': self.model_generator.model.embedding_,
                                                'transformed_embeddings': self.model_generator.model.transform(
                                                    X=self.df[self.features]),
                                                'reconstruction_error': self.model_generator.model.reconstruction_error_
                                                })

    def _multi_dimensional_scaling(self) -> None:
        """
        Multi-dimensional scaling (MDS)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 3})
        else:
            self.kwargs.update({'n_components': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'embeddings': self.model_generator.model.embedding_,
                                                'dissimilarity_matrix': self.model_generator.model.dissimilarity_matrix_,
                                                'stress': self.model_generator.model.stress_,
                                                'n_iter': self.model_generator.model.n_iter_
                                                })

    def _non_negative_matrix_factorization(self) -> None:
        """
        Non-Negative Matrix Factorization for text clustering
        """
        self.cluster[self.ml_algorithm].update({'factorization_matrix_w': self.model_generator.model.transform(X=self.df[self.features]),
                                                'factorization_matrix_h': self.model_generator.model.components_,
                                                'reconstruction_error': self.model_generator.model.reconstruction_err_,
                                                'n_iter': self.model_generator.model.n_iter_
                                                })

    def _ordering_points_to_identify_clustering_structure(self) -> None:
        """
        Ordering points to identify the clustering structure (OPTICS) for clustering complex structures like dense regions in space
        """
        self.cluster[self.ml_algorithm].update({'n_clusters': len(list(set(self.model_generator.model.labels_)))})
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'optics_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'),
                                                             'optics_cluster_partition.html')
            _file_path_cluster_reachability: str = os.path.join(self.kwargs.get('file_path'),
                                                                'optics_cluster_reachability.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
            _file_path_cluster_reachability: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'OPTICS: Silhouette Analysis': dict(data=self.df,
                                                                              features=None,
                                                                              plot_type='silhouette',
                                                                              use_auto_extensions=False if self.kwargs.get(
                                                                                  'use_auto_extensions') is None else self.kwargs.get(
                                                                                  'use_auto_extensions'),
                                                                              file_path=_file_path_silhouette,
                                                                              kwargs=dict(layout={},
                                                                                          n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                          silhouette=_silhouette
                                                                                          )
                                                                              )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'reachability': self.model_generator.model.reachability_,
                                                'ordering': self.model_generator.model.ordering_,
                                                'core_distances': self.model_generator.model.core_distances_,
                                                'predecessor': self.model_generator.model.predecessor_,
                                                'cluster_hierarchy': self.model_generator.model.cluster_hierarchy_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _reachability: pd.DataFrame = pd.DataFrame(
            data={'reachability': self.cluster[self.ml_algorithm].get('reachability')[self.cluster[self.ml_algorithm].get('ordering')],
                  'labels': self.cluster[self.ml_algorithm].get('labels')[self.cluster[self.ml_algorithm].get('ordering')]
                  }
        )
        _reachability = _reachability.replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=False)
        _reachability = _reachability.dropna(axis=0, how='any', inplace=False)
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'OPTICS: Cluster Partition': dict(data=_df,
                                                                    features=self.features,
                                                                    group_by=['cluster'],
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                        'use_auto_extensions') is None else self.kwargs.get(
                                                                        'use_auto_extensions'),
                                                                    file_path=_file_path_cluster_partition,
                                                                    kwargs=dict(layout={})
                                                                    ),
                                  'OPTICS: Reachability': dict(data=_reachability,
                                                               features=['reachability'],
                                                               group_by=['labels'],
                                                               melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                               plot_type='hist',
                                                               use_auto_extensions=False if self.kwargs.get(
                                                                   'use_auto_extensions') is None else self.kwargs.get(
                                                                   'use_auto_extensions'),
                                                               file_path=_file_path_cluster_reachability,
                                                               kwargs=dict(layout={})
                                                               )
                                  })

    def _principal_component_analysis(self) -> None:
        """
        Principal component analysis (PCA)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 2})
        else:
            if self.n_cluster_components >= len(self.features):
                self.kwargs.update({'n_components': 2})
                Log().log(msg='Number of components are greater than or equal to number of features. Number of components are set to 2')
            else:
                self.kwargs.update({'n_components': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'explained_variance_ratio': None,
                                                'cumulative_explained_variance_ratio': None
                                                })
        if self.to_export:
            _file_path_onc: str = os.path.join(self.kwargs.get('file_path'), 'pca_optimal_number_of_components.html')
            _file_path_explained_variance: str = os.path.join(self.kwargs.get('file_path'), 'pca_explained_variance.html')
            _file_path_pca: str = os.path.join(self.kwargs.get('file_path'), 'pca_components.html')
        else:
            _file_path_onc: str = None
            _file_path_explained_variance: str = None
            _file_path_pca: str = None
        if self.find_optimum:
            _cumulative_explained_variance_ratio: np.ndarray = np.cumsum(self.model_generator.model.explained_variance_ratio_)
            _cumulative_variance: pd.DataFrame = pd.DataFrame(data=_cumulative_explained_variance_ratio,
                                                              columns=['cumulative_explained_variance'],
                                                              index=[i for i in
                                                                     range(0, self.kwargs.get('n_components'), 1)]
                                                              )
            _cumulative_variance['component'] = _cumulative_variance.index.values.tolist()
            self.cluster[self.ml_algorithm].update({'explained_variance_ratio': self.model_generator.model.explained_variance_ratio_})
            self.cluster[self.ml_algorithm].update({'cumulative_explained_variance_ratio': _cumulative_explained_variance_ratio})
            self.kwargs.update({'n_components': self._cumulative_explained_variance_ratio(
                explained_variance_ratio=_cumulative_explained_variance_ratio)})
            self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components')})
            self.cluster_plot.update({'PCA: Optimal Number of Components': dict(data=_cumulative_variance,
                                                                                features=['cumulative_explained_variance'],
                                                                                time_features=['component'],
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                    'use_auto_extensions') is None else self.kwargs.get(
                                                                                    'use_auto_extensions'),
                                                                                file_path=_file_path_onc,
                                                                                kwargs=dict(layout={})
                                                                                )
                                      })
            _clustering: PCA = Clustering(cluster_params=self.kwargs).principal_component_analysis()
            _clustering.fit(X=self.df[self.features])
            self.cluster[self.ml_algorithm].update({'fit': _clustering,
                                                    'n_components': self.kwargs.get('n_components')
                                                    })
        _components: pd.DataFrame = pd.DataFrame(data=np.array(self.model_generator.model.components_),
                                                 columns=self.features,
                                                 index=['pc{}'.format(pc) for pc in
                                                        range(0, self.kwargs.get('n_components'), 1)]
                                                 ).transpose()
        _feature_importance: pd.DataFrame = abs(_components)
        self.cluster[self.ml_algorithm].update({'components': self.model_generator.model.components_,
                                                'explained_variance': list(self.model_generator.model.explained_variance_),
                                                'explained_variance_ratio': list(self.model_generator.model.explained_variance_ratio_),
                                                'pc': self.model_generator.model.transform(X=self.df[self.features]),
                                                'feature_importance': dict(names={pca: _feature_importance[pca].sort_values(axis=0, ascending=False).index.values[0] for pca in _feature_importance.keys()},
                                                                           scores=_feature_importance
                                                                           )
                                                })
        for pca in range(0, self.kwargs.get('n_components'), 1):
            self.cluster_plot.update({'PCA: Feature Importance PC{}'.format(pca): dict(data=_feature_importance,
                                                                                       features=None,
                                                                                       plot_type='bar',
                                                                                       use_auto_extensions=False if self.kwargs.get(
                                                                                           'use_auto_extensions') is None else self.kwargs.get(
                                                                                           'use_auto_extensions'),
                                                                                       file_path=os.path.join(self.kwargs.get('file_path'), f'pca_feature_importance_{pca}.html') if self.to_export else None,
                                                                                       kwargs=dict(layout={},
                                                                                                   x=self.features,
                                                                                                   y=_feature_importance[f'pc{pca}'],
                                                                                                   marker=dict(
                                                                                                       color=_feature_importance[f'pc{pca}'],
                                                                                                       colorscale='rdylgn',
                                                                                                       autocolorscale=True
                                                                                                   )
                                                                                                   )
                                                                                       )
                                      })
        self.cluster_plot.update({'PCA: Explained Variance': dict(data=pd.DataFrame(),
                                                                  features=None,
                                                                  plot_type='bar',
                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                      'use_auto_extensions') is None else self.kwargs.get(
                                                                      'use_auto_extensions'),
                                                                  file_path=_file_path_explained_variance,
                                                                  kwargs=dict(layout={},
                                                                              x=list(_feature_importance.keys()),
                                                                              y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                              )
                                                                  ),
                                  'PCA: Principal Components': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('pc'),
                                                                                      columns=list(_feature_importance.keys())
                                                                                      ),
                                                                    features=list(_feature_importance.keys()),
                                                                    plot_type='scatter',
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                        'use_auto_extensions') is None else self.kwargs.get(
                                                                        'use_auto_extensions'),
                                                                    file_path=_file_path_pca,
                                                                    kwargs=dict(layout={},
                                                                                marker=dict(color=self.cluster[self.ml_algorithm].get('pc'),
                                                                                            colorscale='rdylgn',
                                                                                            autocolorscale=True
                                                                                            )
                                                                                )
                                                                    )
                                  })

    def _spectral_clustering(self) -> None:
        """
        Spectral clustering for dimensionality reduction of graphical and non-graphical data
        """
        self.cluster[self.ml_algorithm].update({'n_clusters': len(list(set(self.model_generator.model.labels_)))})
        self.kwargs.update({'n_clusters': self.cluster[self.ml_algorithm].get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'spectral_clustering_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'spectral_clustering_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.labels_)
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'Spectral Clustering: Silhouette Analysis': dict(data=self.df,
                                                                                           features=None,
                                                                                           plot_type='silhouette',
                                                                                           use_auto_extensions=False if self.kwargs.get(
                                                                                               'use_auto_extensions') is None else self.kwargs.get(
                                                                                               'use_auto_extensions'),
                                                                                           file_path=_file_path_silhouette,
                                                                                           kwargs=dict(layout={},
                                                                                                       n_clusters=self.cluster[self.ml_algorithm].get('n_clusters'),
                                                                                                       silhouette=_silhouette
                                                                                                       )
                                                                                           )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'fit': self.model_generator.model,
                                                'affinity_matrix': self.model_generator.model.affinity_matrix_,
                                                'labels': self.model_generator.model.labels_
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('labels')
        self.cluster_plot.update({'Spectral Clustering: Partition': dict(data=_df,
                                                                         features=self.features,
                                                                         group_by=['cluster'],
                                                                         melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                         plot_type='scatter',
                                                                         use_auto_extensions=False if self.kwargs.get(
                                                                             'use_auto_extensions') is None else self.kwargs.get(
                                                                             'use_auto_extensions'),
                                                                         file_path=_file_path_cluster_partition,
                                                                         kwargs=dict(layout={})
                                                                         )
                                  })

    def _spectral_embedding(self) -> None:
        """
        Spectral embedding
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'embeddings': self.model_generator.model.embedding_,
                                                'affinity_matrix': self.model_generator.model.affinity_matrix_
                                                })

    def _t_distributed_stochastic_neighbor_embedding(self) -> None:
        """
        T-distributed stochastic neighbor embedding (TSNE)
        """
        if self.kwargs.get('n_components') is None:
            self.kwargs.update({'n_components': 2})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'embeddings': self.model_generator.model.embedding_
                                                })

    def _truncated_single_value_decomposition(self) -> None:
        """
        Truncated single value decomposition (TSVD / SVD)
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_components': 2})
        else:
            if self.n_cluster_components >= len(self.features):
                self.kwargs.update({'n_components': 2})
                Log().log(msg='Number of components are greater than or equal to number of features. Number of components set to 2')
            else:
                self.kwargs.update({'n_components': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components'),
                                                'explained_variance_ratio': None,
                                                'cumulative_explained_variance_ratio': None
                                                })
        if self.to_export:
            _file_path_onc: str = os.path.join(self.kwargs.get('file_path'), 'svd_optimal_number_of_components.html')
            _file_path_explained_variance: str = os.path.join(self.kwargs.get('file_path'), 'svd_explained_variance.html')
            _file_path_pca: str = os.path.join(self.kwargs.get('file_path'), 'svd_components.html')
        else:
            _file_path_onc: str = None
            _file_path_explained_variance: str = None
            _file_path_pca: str = None
        if self.find_optimum:
            _cumulative_explained_variance_ratio: np.ndarray = np.cumsum(self.model_generator.model.explained_variance_ratio_)
            _cumulative_variance: pd.DataFrame = pd.DataFrame(data=_cumulative_explained_variance_ratio,
                                                              columns=['cumulative_explained_variance'],
                                                              index=[i for i in
                                                                     range(0, self.kwargs.get('n_components'), 1)]
                                                              )
            _cumulative_variance['component'] = _cumulative_variance.index.values.tolist()
            self.cluster[self.ml_algorithm].update(
                {'explained_variance_ratio': self.model_generator.model.explained_variance_ratio_})
            self.cluster[self.ml_algorithm].update(
                {'cumulative_explained_variance_ratio': _cumulative_explained_variance_ratio})
            self.kwargs.update({'n_components': self._cumulative_explained_variance_ratio(
                explained_variance_ratio=_cumulative_explained_variance_ratio)})
            self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components')})
            self.cluster_plot.update({'SVD: Optimal Number of Components': dict(data=_cumulative_variance,
                                                                                features=['cumulative_explained_variance'],
                                                                                time_features=['component'],
                                                                                plot_type='line',
                                                                                use_auto_extensions=False if self.kwargs.get(
                                                                                    'use_auto_extensions') is None else self.kwargs.get(
                                                                                    'use_auto_extensions'),
                                                                                file_path=_file_path_pca,
                                                                                kwargs=dict(layout={})
                                                                                )
                                      })
            self.cluster[self.ml_algorithm].update({'n_components': self.kwargs.get('n_components')})
        _components: pd.DataFrame = pd.DataFrame(data=np.array(self.model_generator.model.components_),
                                                 columns=self.features,
                                                 index=['svd{}'.format(svd) for svd in
                                                        range(0, self.kwargs.get('n_components'), 1)]
                                                 ).transpose()
        _feature_importance: pd.DataFrame = abs(_components)
        self.cluster[self.ml_algorithm].update({'components': self.model_generator.model.components_,
                                                'explained_variance': list(self.model_generator.model.explained_variance_),
                                                'explained_variance_ratio': list(self.model_generator.model.explained_variance_ratio_),
                                                'pc': self.model_generator.model.transform(X=self.df[self.features]),
                                                'feature_importance': dict(
                                                    names={c: _feature_importance[c].sort_values(axis=0, ascending=False).index.values[0]
                                                           for c in _feature_importance.keys()},
                                                    scores=_feature_importance
                                                )
                                                })
        for svd in range(0, self.kwargs.get('n_components'), 1):
            self.cluster_plot.update({f'SVD: Feature Importance PC{svd}': dict(data=_feature_importance,
                                                                               features=None,
                                                                               plot_type='bar',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                   'use_auto_extensions') is None else self.kwargs.get(
                                                                                   'use_auto_extensions'),
                                                                               file_path=os.path.join(self.kwargs.get('file_path'), f'svd_feature_importance_{svd}.html') if self.to_export else None,
                                                                               kwargs=dict(layout={},
                                                                                           x=self.features,
                                                                                           y=_feature_importance[
                                                                                               'svd{}'.format(svd)],
                                                                                           marker=dict(
                                                                                               color=_feature_importance[
                                                                                                   'svd{}'.format(svd)],
                                                                                               colorscale='rdylgn',
                                                                                               autocolorscale=True
                                                                                           )
                                                                                           )
                                                                               )
                                      })
        self.cluster_plot.update({'SVD: Explained Variance': dict(data=pd.DataFrame(),
                                                                  features=None,
                                                                  plot_type='bar',
                                                                  use_auto_extensions=False if self.kwargs.get(
                                                                      'use_auto_extensions') is None else self.kwargs.get(
                                                                      'use_auto_extensions'),
                                                                  file_path=_file_path_explained_variance,
                                                                  kwargs=dict(layout={},
                                                                              x=list(_feature_importance.keys()),
                                                                              y=self.cluster[self.ml_algorithm].get('explained_variance_ratio')
                                                                              )
                                                                  ),
                                  'SVD: Principal Components': dict(data=pd.DataFrame(data=self.cluster[self.ml_algorithm].get('pc'),
                                                                                      columns=list(_feature_importance.keys())
                                                                                      ),
                                                                    features=list(_feature_importance.keys()),
                                                                    melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                    plot_type='scatter',
                                                                    use_auto_extensions=False if self.kwargs.get(
                                                                        'use_auto_extensions') is None else self.kwargs.get(
                                                                        'use_auto_extensions'),
                                                                    file_path=_file_path_pca,
                                                                    kwargs=dict(layout={},
                                                                                marker=dict(color=self.cluster[self.ml_algorithm].get('pc'),
                                                                                            colorscale='rdylgn',
                                                                                            autocolorscale=True
                                                                                            )
                                                                                )
                                                                    )
                                  })

    def _x_means(self) -> None:
        """
        X-Means clustering (partitioning clustering) for graphical data, which is spherical about the cluster centre
        """
        if self.n_cluster_components is None:
            self.kwargs.update({'n_clusters': 2})
        else:
            if self.n_cluster_components < 2:
                Log().log(msg=f"It makes no sense to run cluster analysis with less than 2 clusters ({self.kwargs.get('n_clusters')}). Number of components are set to 2")
                self.kwargs.update({'n_clusters': 2})
            else:
                self.kwargs.update({'n_clusters': self.n_cluster_components})
        self.cluster[self.ml_algorithm].update({'n_clusters': self.kwargs.get('n_clusters')})
        if self.to_export:
            _file_path_silhouette: str = os.path.join(self.kwargs.get('file_path'), 'xmeans_silhouette.html')
            _file_path_cluster_partition: str = os.path.join(self.kwargs.get('file_path'), 'xmeans_cluster_partition.html')
        else:
            _file_path_silhouette: str = None
            _file_path_cluster_partition: str = None
        if self.find_optimum:
            if self.silhouette:
                _silhouette: dict = self.silhouette_analysis(labels=self.model_generator.model.predict(self.df[self.features]))
                self.cluster[self.ml_algorithm].update({'silhouette': _silhouette})
                self.cluster_plot.update({'X-Means: Silhouette Analysis': dict(data=self.df,
                                                                               features=None,
                                                                               plot_type='silhouette',
                                                                               use_auto_extensions=False if self.kwargs.get(
                                                                                   'use_auto_extensions') is None else self.kwargs.get(
                                                                                   'use_auto_extensions'),
                                                                               file_path=_file_path_silhouette,
                                                                               kwargs=dict(layout={},
                                                                                           n_clusters=self.kwargs.get(
                                                                                               'n_clusters'),
                                                                                           silhouette=_silhouette
                                                                                           )
                                                                               )
                                          })
        if 'silhouette' not in self.cluster[self.ml_algorithm].keys():
            self.cluster[self.ml_algorithm].update({'silhouette': None})
        self.cluster[self.ml_algorithm].update({'cluster': self.model_generator.model.predict(X=self.df[self.features]),
                                                'centroids': self.model_generator.model.get_centers(),
                                                'metric': self.model_generator.model.get_total_wce()
                                                })
        _df: pd.DataFrame = self.df
        _df['cluster'] = self.cluster[self.ml_algorithm].get('cluster')
        self.cluster_plot.update({'X-Means: Cluster Partition': dict(data=_df,
                                                                     features=self.features,
                                                                     group_by=['cluster'],
                                                                     melt=True if self.kwargs.get('melt') is None else self.kwargs.get('melt'),
                                                                     plot_type='scatter',
                                                                     use_auto_extensions=False if self.kwargs.get(
                                                                         'use_auto_extensions') is None else self.kwargs.get(
                                                                         'use_auto_extensions'),
                                                                     file_path=_file_path_cluster_partition,
                                                                     kwargs=dict(layout={})
                                                                     )
                                  })

    def main(self, cluster_algorithms: List[str], clean_missing_data: bool = False) -> None:
        """
        Run clustering algorithms

        :param cluster_algorithms: List[str]
            Names of the cluster algorithms

        :param clean_missing_data: bool
            Whether to clean cases containing missing data or not
        """
        _cluster: dict = {}
        _cluster_plot: dict = {}
        if clean_missing_data:
            self._clean_missing_data()
        for cl in cluster_algorithms:
            self.ml_algorithm = cl
            self.cluster.update({cl: {}})
            ################################
            # Principal Component Analysis #
            ################################
            if cl == 'pca':
                self._principal_component_analysis()
            ###################
            # Factor Analysis #
            ###################
            elif cl in ['fa', 'factor']:
                self._factor_analysis()
            ########################################
            # Truncated Single Value Decomposition #
            ########################################
            elif cl in ['svd', 'tsvd']:
                self._truncated_single_value_decomposition()
            ###############################################
            # t-Distributed Stochastic Neighbor Embedding #
            ###############################################
            elif cl == 'tsne':
                self._t_distributed_stochastic_neighbor_embedding()
            #############################
            # Multi Dimensional Scaling #
            #############################
            elif cl == 'mds':
                self._multi_dimensional_scaling()
            #####################
            # Isometric Mapping #
            #####################
            elif cl == 'isomap':
                self._isometric_mapping()
            ######################
            # Spectral Embedding #
            ######################
            elif cl in ['spectral_emb', 'spectral_embedding']:
                self._spectral_embedding()
            ############################
            # Locally Linear Embedding #
            ############################
            elif cl in ['lle', 'locally_emb', 'locally_linear', 'locally_embedding']:
                self._locally_linear_embedding()
            ###########
            # K-Means #
            ###########
            elif cl == 'kmeans':
                self._k_means()
            #############
            # K-Medians #
            #############
            elif cl == 'kmedians':
                self._k_medians()
            #############
            # K-Medoids #
            #############
            elif cl == 'kmedoids':
                self._k_medoids()
            #####################################
            # Non-Negative Matrix Factorization #
            #####################################
            elif cl == 'nmf':
                self._non_negative_matrix_factorization()
            ###############################
            # Latent Dirichlet Allocation #
            ###############################
            elif cl == 'lda':
                self._latent_dirichlet_allocation()
            ########################################################
            # Ordering Points To Identify the Clustering Structure #
            ########################################################
            elif cl == 'optics':
                self._ordering_points_to_identify_clustering_structure()
            ###############################################################
            # Density-Based Spatial Clustering of Applications with Noise #
            ###############################################################
            elif cl == 'dbscan':
                self._density_based_spatial_clustering_applications_with_noise()
            #######################
            # Spectral Clustering #
            #######################
            elif cl in ['spectral_cl', 'spectral_cluster']:
                self._spectral_clustering()
            #########################
            # Feature Agglomeration #
            #########################
            elif cl in ['feature_agglo', 'feature_agglomeration']:
                self._feature_agglomeration()
            ############################
            # Agglomerative Clustering #
            ############################
            elif cl in ['agglo_cl', 'agglo_cluster', 'struc_agglo_cl', 'struc_agglo_cluster', 'unstruc_agglo_cl', 'unstruc_agglo_cluster']:
                self._agglomerative_clustering()
            ####################
            # Birch Clustering #
            ####################
            elif cl == 'birch':
                self._birch_clustering()
            ########################
            # Affinity Propagation #
            ########################
            elif cl in ['affinity_prop', 'affinity_propagation']:
                self._affinity_propagation()
            ###########
            # X-Means #
            ###########
            elif cl == 'xmeans':
                self._x_means()
            else:
                raise UnsupervisedMLException(f'Clustering algorithm ({cl}) not supported')

    def silhouette_analysis(self, labels: List[int]) -> dict:
        """
        Calculate silhouette scores to evaluate optimal amount of clusters for most cluster analysis algorithms

        :param labels: List[int]
            Predicted cluster labels by any cluster algorithm

        :return: dict
            Optimal clusters and the average silhouette score as well as silhouette score for each sample
        """
        _lower: int = 10
        _silhouette: dict = {}
        _avg_silhoutte_score: List[float] = []
        if self.kwargs.get('n_clusters') is None:
            if self.n_cluster_components is None:
                self.kwargs.update({'n_clusters': 2})
            else:
                if self.n_cluster_components < 2:
                    Log(write=False, level='info').log(
                        msg='It makes no sense to run cluster analysis with less than 2 clusters ({}). Run analysis with more than 1 cluster instead'.format(
                            self.kwargs.get('n_clusters')))
                    self.kwargs.update({'n_clusters': 2})
                else:
                    self.kwargs.update({'n_clusters': self.n_cluster_components})
        _clusters: List[int] = [n for n in range(0, self.kwargs.get('n_clusters'), 1)]
        for cl in _clusters:
            _avg_silhoutte_score.append(silhouette_score(X=self.df[self.features],
                                                         labels=labels,
                                                         metric='euclidean' if self.metric is None else self.metric,
                                                         sample_size=self.kwargs.get('sample_size'),
                                                         random_state=self.seed
                                                         )
                                        )
            _silhouette.update({'cluster_{}_avg'.format(cl): _avg_silhoutte_score[-1]})
            _silhouette_samples: np.array = silhouette_samples(X=self.df[self.features],
                                                               labels=labels,
                                                               metric='euclidean' if self.metric is None else self.metric,
                                                               )
            for s in range(0, cl + 1, 1):
                _s: np.array = _silhouette_samples[labels == s]
                _s.sort()
                _upper: int = _lower + _s.shape[0]
                _silhouette.update({'cluster_{}_samples'.format(cl): dict(y=np.arange(_lower, _upper), scores=_s)})
                _lower = _upper + 10
        _max_avg_score: float = max(_avg_silhoutte_score)
        _silhouette.update({'best': dict(cluster=_avg_silhoutte_score.index(_max_avg_score) + 1, avg_score=_max_avg_score)})
        return _silhouette
