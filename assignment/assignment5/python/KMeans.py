import random
from sys import maxsize
from math import sqrt, inf

RANDOM_SEED = 1208
LARGE_NUMBER = inf
EQUAL_THRESHOLD = 0.0001

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def initialCentroidsRandomPick(self, X, k):
        """
        :return centroids: dict - {0:[], tag: centroid_coordinate, ...}
        """
        X_length = len(X)
        random.seed(RANDOM_SEED)
        idx_centroids = random.sample(range(X_length), k)
#         centroids = [X[idx] for idx in idx_centroids]
        centroids = {i: X[idx] for i, idx in enumerate(idx_centroids)}
        return centroids
    
    def computeDistance(self, v1, v2):
        # Euclidean Distance
        pingfanghe = 0
        for i in range(len(v1)):
            pingfanghe += (v1[i] - v2[i])**2
        return sqrt(pingfanghe)

    def clusterPoints(self, X, centroids):
        """
        :return clusters: {cluster_idx: [point_idx, ...], ...}
        """
        clusters = {}
        for i, x in enumerate(X):
            min_distance = (0, LARGE_NUMBER)
            for k in centroids:
                dis = self.computeDistance(x, centroids[k])
                if dis < min_distance[1]:
                    min_distance = (k, dis)
            cid = min_distance[0]
            if clusters.get(cid) == None:
                clusters[cid] = [i]
            else:
                clusters[cid].append(i)
        return clusters
    
    def computeCentroids(self, X, clusters):
        dimension = len(X[0])
        new_centroids = {}
        for cid in clusters:
            cluster = clusters[cid]
            n = len(cluster)
            centroid = [0] * dimension
            for idx in cluster:
                point = X[idx]
                for i, v in enumerate(point):
                    centroid[i] += v
            for i, v in enumerate(centroid):
                centroid[i] = v / n
            new_centroids[cid] = centroid
        return new_centroids
    
    def checkClustersChanged(self, clusters, new_clusters):
        if clusters == None:
            return True
        for k in clusters:
            set1 = set(clusters[k])
            set2 = set(new_clusters[k])
            if set1 != set2:
                return True
        return False
    
    def fit(self, X):
        # when all of clusters don't change, stop
        """
        :param X: [[]] - matrix
        """
        # initial centroids
        centroids = self.initialCentroidsRandomPick(X, self.n_clusters)
        # cluster points
        clusters = None
        new_clusters = self.clusterPoints(X, centroids)
        
        num_iteration = 0
        while(self.checkClustersChanged(clusters, new_clusters)):
            # compute centroids
            new_centroids = self.computeCentroids(X, new_clusters)
            
            clusters = new_clusters
            # cluster points
            new_clusters = self.clusterPoints(X, new_centroids)
            num_iteration += 1
#             print(num_iteration)
        print('number_of_iterations:', num_iteration) 
        return new_clusters
    
    def checkCentroidsChanged(self, centroids, new_centroids):
        if centroids == None:
            return True
        for k in centroids:
            centroid = centroids[k]
            new_centroid = new_centroids[k]
            for i in range(len(centroid)):
                if abs(centroid[i] - new_centroid[i]) > EQUAL_THRESHOLD:
                    return True
        return False     
        
    def fit2(self, X):
        # when all of the positions of centroids don't change, stop
        """
        :param X: [[]] - matrix
        """
        # initial centroids
        centroids = None
        new_centroids = self.initialCentroidsRandomPick(X, self.n_clusters)
        # cluster points
        clusters = self.clusterPoints(X, new_centroids)
        
        # num_iteration = 1
        while(self.checkCentroidsChanged(centroids, new_centroids)):
            # print(num_iteration)
            # num_iteration += 1
            centroids = new_centroids
            # compute centroids
            new_centroids = self.computeCentroids(X, clusters)

            # cluster points
            clusters = self.clusterPoints(X, new_centroids)
            
        return clusters
