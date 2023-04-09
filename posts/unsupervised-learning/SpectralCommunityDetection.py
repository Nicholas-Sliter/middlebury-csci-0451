import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class SpectralCommunityDetection:

    @staticmethod
    def get_nearest_neighbor_graph(X, k):
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        _, indices = nbrs.kneighbors(X)
        for i in range(len(X)):
            for j in indices[i]:
                G.add_edge(i, j)
        return G
    

    @staticmethod
    def get_adjacency_matrix(G):
        return nx.adjacency_matrix(G).toarray()
    
    @classmethod
    def get_laplacian_matrix(cls, G):
        A = cls.get_adjacency_matrix(G)
        return np.diag(A.sum(axis=0)) - A

        
    @staticmethod
    def get_eigenvectors(L, k):
        e_values, e_vectors = np.linalg.eig(L) 

        e_values = np.real(e_values) # We need to discard imaginary parts for kmeans
        e_vectors = np.real(e_vectors)

        idx = np.argsort(e_values)[:k]
        return e_vectors[:, idx]
    
    @staticmethod
    def k_means_clustering(X, k):
        kmeans = KMeans(n_clusters=k).fit(X)
        return kmeans.labels_


    @classmethod
    def get_clusters(cls, G, k):
        L = cls.get_laplacian_matrix(G)
        X = cls.get_eigenvectors(L, k)
        return cls.k_means_clustering(X, k)
    
    @classmethod
    def spectral_clustering(cls, G, k):
        clusters = cls.get_clusters(G, k)
        return cls.build_community_label_vector(clusters, k)
    
    @classmethod
    def get_multiway_clusters(cls, G, k):
        return cls.spectral_clustering(G, k)

    @staticmethod
    def build_community_label_vector(clusters, k):
        # [[0, 1, 2], [3, 4, 5]] -> [0, 0, 0, 1, 1, 1]
        community_labels = np.zeros(len(clusters)).astype(int)
        for i in range(k):
            community_labels[clusters == i] = i
        return community_labels
