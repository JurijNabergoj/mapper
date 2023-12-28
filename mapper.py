import gudhi
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import seaborn as sns


def cluster(data, algorithm="DBSCAN", dbscan_eps=4.4, dbscan_min_samples=5):
    algorithms = {"DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
                  "AC": AgglomerativeClustering()}
    return algorithms[algorithm].fit_predict(data)


def read3d(path):
    plydata = PlyData.read(path)
    data = np.array([list(v) for v in plydata.elements[0].data])
    return data


def store_clusters(clusters, cluster_indices, all_clusters, all_cluster_indices):
    clusters_inds = np.unique(clusters)
    for idx in range(len(clusters_inds)):
        indices = [i for i, value in enumerate(clusters) if value == idx]
        cluster_idxs = cluster_indices[indices]
        cluster = data[cluster_idxs]
        all_cluster_indices.append(cluster_idxs)
        all_clusters.append(cluster)
    return


def plot_interval_points(pts, colors, order=None):
    if order is None:
        order = (1, 2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, partition in enumerate(pts):
        partition_array = np.array(partition)
        x = partition_array[:, order[0]]
        y = partition_array[:, order[1]]
        z = partition_array[:, order[2]]
        ax.scatter(x, y, z, label=f"Partition {i + 1}", color=colors[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_interval_points_with_clusters(
        pts, cluster_labels, colors, order=None, markers=None
):
    if order is None:
        order = (0, 1, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    c = 0
    for i, (partition, labels) in enumerate(zip(pts, cluster_labels)):
        partition_array = np.array(partition)
        for j, cluster_label in enumerate(np.unique(labels)):
            cluster_points = partition_array[labels == cluster_label]

            ax.scatter(
                cluster_points[:, order[0]],
                cluster_points[:, order[1]],
                cluster_points[:, order[2]],
                label=f"Partition {i + 1}, Cluster {cluster_label + 1}",
                color=colors[c],
                marker=markers[c] if markers else None,
            )
            c += 1
    ax.set_title("Object 3D plot with partitions and clusters")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend(prop={"size": 7})
    plt.tight_layout()
    plt.show()


def plot_mapper(graph, palette):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pos = nx.spring_layout(graph, dim=3)
    for node, (x, y, z) in pos.items():
        ax.scatter(x, y, z, color=palette[node - 1], s=50, label=str(node))
    for edge in graph.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color='black')
    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, str(node), fontsize=10, color="black")

    plt.title("Mapper graph")
    plt.show()


def find_triangles(edges):
    triangles = []
    for i in range(len(edges)):
        edge1 = edges[i]
        for j in range(i + 1, len(edges)):
            edge2 = edges[j]
            if edge1 != edge2:
                common_vertex12 = set(edge1) & set(edge2)
                v1 = (set(edge1) - common_vertex12).pop()
                v2 = (set(edge2) - common_vertex12).pop()
                if len(common_vertex12) == 1:
                    for k in range(j + 1, len(edges)):
                        edge3 = edges[k]
                        if edge1 != edge3 and edge2 != edge3:
                            if v1 in set(edge3) and v2 in set(edge3):
                                triangle = [common_vertex12.pop(), v1, v2]
                                if triangle not in triangles:
                                    triangles.append(triangle)
    return triangles


# MEASUREMENT FUNCTIONS


# returns function which returns which takes a point cloud and returns given axis values
def axis(axis):
    return lambda data: data[:, axis]


# reduces point cloud to 1 dim with PCA
def pca(data):
    return PCA(n_components=1).fit_transform(data)


# reduces point cloud to 1 dim with t-SNE
def t_sne(data):
    return TSNE(n_components=1).fit_transform(data)


# returns distances of every point to centroid
def centroid_dist(data):
    centroid = np.mean(data, axis=0)
    diff = data - centroid
    return np.sqrt(
        np.power(diff[:, 0], 2) + np.power(diff[:, 1], 2) + np.power(diff[:, 2], 2)
    )


# applies given measurement function to given point cloud and returns measurements for every point
def apply_measurement_function(data, function="axis0"):
    functions = {
        "axis0": axis(0),
        "axis1": axis(1),
        "axis2": axis(2),
        "PCA": pca,
        "t-SNE": t_sne,
        "radial": centroid_dist,
    }
    return functions[function](data)


# partitions data into n groups based on measures of points
def partition(data, measures, n=None, overlap=0.2):
    n = int(np.log(data.shape[0])) if n == None else n

    max_ = np.max(measures)
    min_ = np.min(measures)

    overlap_size = overlap * (max_ - min_) / n

    intervals = np.linspace(min_, max_, n + 1)
    partitions = [None] * n
    partition_indices = [None] * n
    for i in range(n):
        a = intervals[i]
        b = intervals[i + 1]
        points = []
        indices = []
        for index, (point, measure) in enumerate(zip(data, measures)):
            if a - overlap_size < measure < b + overlap_size:
                points.append(point)
                indices.append(index)
        partitions[i] = np.array(points)
        partition_indices[i] = np.array(indices)

    return partitions, partition_indices


# plots partitioned data
def plot_partitions(partitions):
    ax = plt.axes(projection="3d")
    for p in partitions:
        x = p.T[0]
        z = p.T[1]
        y = p.T[2]

        # plotting
        ax.scatter(x, y, z, alpha=0.5)
        ax.set_title("Partitioned 3D object")
    plt.show()


# returns partitions of point cloud with given measurement function
def partition_data(
        data, measurement_function="PCA", n_partitions=None, overlap=0.2, plot=False
):
    measures = apply_measurement_function(data, function=measurement_function)
    partitions, indices = partition(data, measures, n=n_partitions, overlap=overlap)
    if plot:
        plot_partitions(partitions)
    return partitions, indices


# plot 3d object
def plot3d(data):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x = data.T[0]
    z = data.T[1]
    y = data.T[2]

    ax.scatter(x, y, z, "green")
    ax.set_title("3D plot")
    plt.show()


def compute_clusters(partitions, partition_indices):
    all_clusters = []
    all_cluster_indices = []
    cluster_sets = []
    for i in range(len(partitions)):
        partition_points = partitions[i]
        curr_clusters = cluster(partition_points, algorithm="AC")
        store_clusters(curr_clusters, partition_indices[i], all_clusters, all_cluster_indices)
        cluster_sets.append(curr_clusters)
    return all_clusters, all_cluster_indices, cluster_sets


def get_colors_palettes(num_partitions, cluster_sets):
    palette = sns.color_palette("husl", n_colors=num_partitions)
    occurrences = [len(np.unique(cls)) for cls in cluster_sets]
    clrs_indices = np.repeat(np.arange(len(occurrences)), occurrences)
    clrs = [palette[i] for i in clrs_indices]
    cluster_palette = sns.color_palette("husl", n_colors=len(clrs))
    return palette, cluster_palette


def generate_graph(all_clusters, all_cluster_indices):
    # Create Mapper graph
    # - each cluster is a node
    # - two nodes are connected if their clusters share points
    graph = nx.Graph()
    for i in range(len(all_cluster_indices)):
        centroid = np.mean(all_clusters[i], axis=0)
        graph.add_node(i, pos=centroid)
        if i > 0:
            for j in range(i):
                if len(set(all_cluster_indices[i]) & set(all_cluster_indices[j])) > 0:
                    graph.add_edge(i, j)
    return graph


if __name__ == "__main__":
    data = read3d("data/table.ply")

    # reduce density
    data = data[::20]
    plot3d(data)

    # possible measurement functions: 'axis0', 'axis1', 'axis2', 'PCA', 't-SNE', 'radial'
    partitions, partition_indices = partition_data(
        data, measurement_function="axis0", n_partitions=None, overlap=0.2, plot=True
    )
    # Compute clusters for each partition
    all_clusters, all_cluster_indices, cluster_sets = compute_clusters(partitions, partition_indices)

    # Get colors for plotting partitions and clusters
    palette, cluster_palette = get_colors_palettes(len(partitions), cluster_sets)

    # 3d partition plot with specified order of axes
    plot_interval_points(partitions, palette, order=(0, 1, 2))
    plot_interval_points(partitions, palette, order=(1, 0, 2))

    # 3d cluster plot with specified order of axes
    plot_interval_points_with_clusters(partitions, cluster_sets, cluster_palette, order=(0, 1, 2))
    plot_interval_points_with_clusters(partitions, cluster_sets, cluster_palette, order=(1, 0, 2))

    # Generate Mapper graph
    graph = generate_graph(all_clusters, all_cluster_indices)

    # Plot Mapper graph
    plot_mapper(graph, cluster_palette)

    # UNFINISHED
    '''
    triangles = find_triangles(list(graph.edges))
    
    # Convert to gudhi simplex tree
    simplex_tree = gudhi.SimplexTree()
    for simplex in graph.edges:
        simplex_tree.insert(simplex)

    simplices = [s[0] for s in list(simplex_tree.get_skeleton(2))]
    
    # Compute persistent homology
    persistence = simplex_tree.persistence(min_persistence=0.01)
    diagrams = gudhi.plot_persistence_diagram(persistence)
    plt.show()
    
    # Plot persistent homology diagram
    gudhi.plot_persistence_barcode(persistence)
    plt.show()
    '''
