import gudhi
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import seaborn as sns
import math


def cluster(data, algorithm="DBSCAN", dbscan_eps=5, dbscan_min_samples=10):
    if data.shape[0] == 0:
        return np.array([])

    algorithms = {
        "DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
        "AC": AgglomerativeClustering(),
        "AC2": AgglomerativeClustering(
            n_clusters=None, distance_threshold=10, compute_full_tree=True
        ),
    }
    return algorithms[algorithm].fit_predict(data)


def read3d(path, density=1, resize=1):
    plydata = PlyData.read(path)
    data = resize * (
        np.array([list(v) for v in plydata.elements[0].data])[:: int(1 / density)]
    )
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
        if partition.shape[0] != 0:
            partition_array = np.array(partition)
            x = partition_array[:, order[0]]
            y = partition_array[:, order[1]]
            z = partition_array[:, order[2]]
            ax.scatter(x, y, z, label=f"Partition {i + 1}", color=colors[i])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Partitioned 3D Points")
    # ax.legend()

    plt.tight_layout()
    set_axes_equal(ax)
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
    # plt.legend(prop={"size": 7})
    plt.tight_layout()
    set_axes_equal(ax)
    plt.show()


def plot_mapper(graph, pos, palette):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for node, (x, y, z) in pos.items():
        ax.scatter(x, y, z, color=palette[node - 1], s=50, label=str(node))
    for edge in graph.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, color="black")
    for node, (x, y, z) in pos.items():
        ax.text(x, y, z, str(node), fontsize=10, color="black")

    plt.title("Mapper graph")
    set_axes_equal(ax)
    plt.show()


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
        # if parition is empty then pass
        if p.shape[0] == 0:
            pass
        else:
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


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# plot 3d object
def plot3d(data):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    x = data.T[0]
    z = data.T[1]
    y = data.T[2]

    ax.scatter(x, y, z, "green")
    ax.set_title("3D plot")
    set_axes_equal(ax)
    plt.show()


def compute_clusters(partitions, partition_indices, algorithm="DBSCAN"):
    all_clusters = []
    all_cluster_indices = []
    cluster_sets = []
    for i in range(len(partitions)):
        partition_points = partitions[i]
        curr_clusters = cluster(partition_points, algorithm=algorithm)
        store_clusters(
            curr_clusters, partition_indices[i], all_clusters, all_cluster_indices
        )
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
    positions = {}
    for i in range(len(all_cluster_indices)):
        centroid = np.mean(all_clusters[i], axis=0)
        positions[i] = np.array(centroid)
        graph.add_node(i, pos=centroid)
        if i > 0:
            for j in range(i):
                if len(set(all_cluster_indices[i]) & set(all_cluster_indices[j])) > 0:
                    graph.add_edge(i, j)
    return graph, positions


def twin_torus():
    m = 50
    n = 20
    distance = 10
    r_big = distance / 2
    r_small = 2
    centre1 = np.array([-distance / 2, 0, 0])
    centre2 = np.array([distance / 2, 0, 0])
    torus1 = []
    torus2 = []
    for i in range(m):
        phi = i / m * 2 * math.pi
        for j in range(n):
            psi = j / n * 2 * math.pi
            x = r_big * math.cos(phi) + r_small * math.cos(phi) * math.cos(psi)
            y = r_big * math.sin(phi) + r_small * math.sin(phi) * math.cos(psi)
            z = r_small * math.sin(psi)
            # print(phi, psi)
            # print(x, y, z)
            if True:
                torus1.append(centre1 + np.array([x, y, z]))
            if True:
                torus2.append(centre2 + np.array([x, y, z]))
        # print("\n")
    torus1.extend(torus2)
    return np.array(torus1)


def k_torus(k=1):
    m = 50
    n = 15
    distance = 10
    r_big = distance / 2
    r_small = 2
    centres = np.array([[distance * i, 0, 0] for i in range(k)])
    tori = [[]] * n
    for i in range(m):
        phi = i / m * 2 * math.pi
        for j in range(n):
            psi = j / n * 2 * math.pi
            x = r_big * math.cos(phi) + r_small * math.cos(phi) * math.cos(psi)
            y = r_big * math.sin(phi) + r_small * math.sin(phi) * math.cos(psi)
            z = r_small * math.sin(psi)
            # print(phi, psi)
            # print(x, y, z)
            for l in range(k):
                tori[l].append(centres[l] + np.array([x, y, z]))
        # print("\n")
    joined = []
    for i in range(k):
        joined = joined + tori[i]
    return np.array(joined)


def join_shapes(shape1, shape2, loc1, loc2):
    shape1 = loc1 + shape1
    shape2 = loc2 + shape2
    return np.concatenate((shape1, shape2))


if __name__ == "__main__":
    table = read3d("data/table.ply", density=0.2, resize=0.15)

    data = join_shapes(k_torus(1), table, np.array([0, -10, 0]), np.array([0, 10, 0]))
    # reduce density
    # data = data[::1]
    plot3d(data)

    # possible measurement functions: 'axis0', 'axis1', 'axis2', 'PCA', 't-SNE', 'radial'
    partitions, partition_indices = partition_data(
        data, measurement_function="PCA", n_partitions=40, overlap=0.15, plot=True
    )
    # Compute clusters for each partition
    all_clusters, all_cluster_indices, cluster_sets = compute_clusters(
        partitions, partition_indices, algorithm="AC"
    )

    # Get colors for plotting partitions and clusters
    palette, cluster_palette = get_colors_palettes(len(partitions), cluster_sets)

    # 3d partition plot with specified order of axes
    plot_interval_points(partitions, palette, order=(0, 1, 2))

    # 3d cluster plot with specified order of axes
    plot_interval_points_with_clusters(
        partitions, cluster_sets, cluster_palette, order=(0, 1, 2)
    )
    # Generate Mapper graph
    graph, centroid_positions = generate_graph(all_clusters, all_cluster_indices)

    # Plot Mapper graph
    plot_mapper(graph, centroid_positions, cluster_palette)

    # Convert to gudhi simplex tree
    simplex_tree = gudhi.SimplexTree()
    for simplex in graph.edges:
        simplex_tree.insert(simplex)

    # Compute persistent homology
    persistence = simplex_tree.persistence(persistence_dim_max=True, min_persistence=-1.0)
    diagrams = gudhi.plot_persistence_diagram(persistence)
    plt.show()

    # Plot persistent homology diagram
    gudhi.plot_persistence_barcode(persistence)
    plt.show()
