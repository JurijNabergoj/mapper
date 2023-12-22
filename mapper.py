import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
import seaborn as sns


def filter(data, lens="PCA"):
    lenses = {"PCA": PCA, "t-SNE": TSNE}
    return lenses[lens](n_components=2).fit_transform(data)


def cluster(data, algorithm="DBSCAN"):
    algorithms = {"DBSCAN": DBSCAN}
    return algorithms[algorithm].fit(data)


def read3d(path):
    plydata = PlyData.read(path)
    data = np.array([list(v) for v in plydata.elements[0].data])
    return data


def make_partitions(clusters, inds, all_parts):
    num_partitions = len(np.unique(clusters))
    partitions = [[] for _ in range(num_partitions)]
    for i, idx in enumerate(inds):
        partition_index = clusters[i]
        partitions[partition_index].append(idx)
    partitions = [np.array(partition) for partition in partitions]
    for part in partitions:
        all_parts.append(part)
    return


def get_intervals(projected_data, overlap_factor):
    min_val, max_val = np.min(projected_data[:, 1]), np.max(projected_data[:, 1])
    interval_width = (max_val - min_val) / num_intervals
    overlap = int(overlap_factor * interval_width)
    intervals = [(min_val + i * interval_width, min_val + (i + 1) * interval_width + overlap) for i in
                 range(num_intervals)]
    return intervals


def plot_interval_points(interval_pts, colors, order=None):
    if order is None:
        order = (1, 2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, partition in enumerate(interval_pts):
        partition_array = np.array(partition)
        x = partition_array[:, order[0]]
        y = partition_array[:, order[1]]
        z = partition_array[:, order[2]]
        ax.scatter(x, y, z, label=f'Partition {i + 1}', color=colors[i])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points')

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_interval_points_with_clusters(interval_pts, cluster_labels, colors, order=None, markers=None):
    if order is None:
        order = (0, 1, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = 0
    for i, (partition, labels) in enumerate(zip(interval_pts, cluster_labels)):
        partition_array = np.array(partition)
        for j, cluster_label in enumerate(np.unique(labels)):
            cluster_points = partition_array[labels == cluster_label]

            ax.scatter(cluster_points[:, order[0]], cluster_points[:, order[1]], cluster_points[:, order[2]],
                       label=f'Partition {i + 1}, Cluster {cluster_label + 1}',
                       color=colors[c], marker=markers[c] if markers else None)
            c += 1
    ax.set_title('Object 3D plot with partitions and clusters')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data3d = read3d("data/table.ply")

    # reduce density
    data3d = data3d[::10]
    fig = plt.figure()

    # syntax for 3-D projection
    ax = plt.axes(projection="3d")

    # defining all 3 axes
    z = data3d.T[0]
    x = data3d.T[1]
    y = data3d.T[2]

    # plotting
    ax.scatter(x, y, z, "green")
    ax.set_title("Object 3D plot")
    plt.show()

    # get 2d data
    data2d = filter(data3d, lens="t-SNE")

    # plot 2d
    plt.scatter(data2d[:, 0], data2d[:, 1])
    plt.title("Object 2D plot")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    # create intervals along height (y-axis) of data projected to 2d
    num_intervals = 3
    intervals = get_intervals(data2d, overlap_factor=0.1)

    # create palette of colors for function 'plot_interval_points'
    palette = sns.color_palette("husl", n_colors=num_intervals)

    # for points in each interval compute clusters and store the computed partitions in 'all_partitions'
    # very important is the selection of eps and min_samples when using DBSCAN.
    interval_pts = []
    all_partitions = []
    cluster_sets = []
    for i in range(num_intervals):
        interval_mask = (data2d[:, 1] >= intervals[i][0]) & (data2d[:, 1] < intervals[i][1])
        interval_indices = [index for index, value in enumerate(interval_mask) if value]
        interval_points = data3d[interval_mask]
        clusters = DBSCAN(eps=4.4, min_samples=5).fit_predict(interval_points)
        make_partitions(clusters, interval_indices, all_partitions)

        interval_pts.append(interval_points)
        cluster_sets.append(clusters)

    # calculate number of colors to be used in the cluster_palette
    # cluster palette is used in 'plot_interval_points_with_clusters'
    occurences = [len(np.unique(cls)) for cls in cluster_sets]
    colors = np.repeat(np.arange(len(occurences)), occurences)
    clrs = [palette[i] for i in colors]
    cluster_palette = sns.color_palette("husl", n_colors=len(clrs))

    # 3d plotting functions with specified order of axes.
    plot_interval_points(interval_pts, palette, order=(0, 1, 2))
    plot_interval_points(interval_pts, palette, order=(1, 0, 2))
    plot_interval_points_with_clusters(interval_pts, cluster_sets, cluster_palette, order=(0, 1, 2))
    plot_interval_points_with_clusters(interval_pts, cluster_sets, cluster_palette, order=(1, 0, 2))

    # plot Mapper graph
    # - each cluster is a node
    # - two nodes are connected if their clusters share points
    graph = nx.Graph()
    for i in range(len(all_partitions)):
        graph.add_node(i)
        if i > 0:
            for j in range(i):
                if len(set(all_partitions[i]) & set(all_partitions[j])) > 0:
                    graph.add_edge(i, j)

    plt.title('Mapper graph')
    nx.draw_spring(graph, with_labels=True, node_color=cluster_palette, font_weight='bold', node_size=500,
                   font_color='black',
                   font_size=10)
    plt.show()
