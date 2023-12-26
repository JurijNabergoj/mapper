import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import seaborn as sns
import gudhi
import itertools


def filter(data, lens="PCA"):
    lenses = {"PCA": PCA, "t-SNE": TSNE}
    return lenses[lens](n_components=2).fit_transform(data)


def cluster(data, algorithm="DBSCAN", dbscan_eps=4.4, dbscan_min_samples=5, **kwargs):
    algorithms = {"DBSCAN": DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples),
                  "AC": AgglomerativeClustering()}
    return algorithms[algorithm].fit_predict(data)


def read3d(path):
    plydata = PlyData.read(path)
    data = np.array([list(v) for v in plydata.elements[0].data])
    return data


def make_partitions(clusters, indices, all_parts):
    num_partitions = len(np.unique(clusters))
    partitions = [[] for _ in range(num_partitions)]
    for i, idx in enumerate(indices):
        partition_index = clusters[i]
        partitions[partition_index].append(idx)
    partitions = [np.array(partition) for partition in partitions]
    for part in partitions:
        all_parts.append(part)
    return


def generate_intervals(projected_data, overlap_factor):
    min_val, max_val = np.min(projected_data[:, 1]), np.max(projected_data[:, 1])
    interval_width = (max_val - min_val) / num_intervals
    overlap = int(overlap_factor * interval_width)
    intervals = [(min_val + i * interval_width, min_val + (i + 1) * interval_width + overlap) for i in
                 range(num_intervals)]
    return intervals


def plot_interval_points(pts, colors, order=None):
    if order is None:
        order = (1, 2, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, partition in enumerate(pts):
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


def plot_interval_points_with_clusters(pts, cluster_labels, colors, order=None, markers=None):
    if order is None:
        order = (0, 1, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = 0
    for i, (partition, labels) in enumerate(zip(pts, cluster_labels)):
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


def plot_mapper(graph, palette):
    nx.draw_spring(graph, with_labels=True, node_color=palette, font_weight='bold', node_size=500,
                   font_color='black',
                   font_size=10)
    plt.title('Mapper graph')
    plt.show()


def draw_2d_simplicial_complex(simplices, pos=None, return_pos=False, ax=None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].

        Args
        ----
        simplices: list of lists of integers
            List of simplices to draw. Sub-simplices are not needed (only maximal).
            For example, the 2-simplex [1,2,3] will automatically generate the three
            1-simplices [1,2],[2,3],[1,3] and the three 0-simplices [1],[2],[3].
            When a higher order simplex is entered only its sub-simplices
            up to D=2 will be drawn.

        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.

        ax: matplotlib.pyplot.axes (default=None)

        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.

        References
        ----------
        . [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2019)
               "Simplicial Models of Social Contagion".
               Nature communications, 10(1), 2485.
    """

    # List of 0-simplices
    nodes = list(set(itertools.chain(*simplices)))

    # List of 1-simplices
    edges = list(set(itertools.chain(
        *[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    # List of 2-simplices
    triangles = list(set(itertools.chain(
        *[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))

    if ax is None: ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.axis('off')

    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)

    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([x0, x1], [y0, y1], color='black', zorder=1, lw=0.7)
        ax.add_line(line)

    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([[x0, y0], [x1, y1], [x2, y2]],
                          edgecolor='black', facecolor=plt.cm.Blues(0.6),
                          zorder=2, alpha=0.4, lw=0.5)
        ax.add_patch(tri)

    # Drawing the nodes
    for i in nodes:
        (x, y) = pos[i]
        circ = plt.Circle([x, y], radius=0.02, zorder=3, lw=0.5,
                          edgecolor='Black', facecolor=u'#ff7f0e')
        ax.add_patch(circ)

    if return_pos:
        return pos


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


if __name__ == '__main__':
    data3d = read3d("data/table.ply")
    # Reduce density
    data3d = data3d[::10]

    # data3d = generate_torus_points(radius_major=10.0, radius_minor=5.0, num_points=1000)

    fig = plt.figure()
    # Syntax for 3-D projection
    ax = plt.axes(projection="3d")

    # Defining all 3 axes
    z = data3d.T[0]
    x = data3d.T[1]
    y = data3d.T[2]

    # Plotting 3d object
    ax.scatter(x, y, z, "green")
    ax.set_title("Object 3D plot")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

    # Get 2d data using filter
    data2d = filter(data3d, lens="t-SNE")

    # Plot 2d object
    plt.scatter(data2d[:, 0], data2d[:, 1])
    plt.title("Object 2D plot")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

    num_intervals = 5
    overlap_factor = 0.4

    # Create intervals along height (y-axis) of data projected to 2d
    intervals = generate_intervals(data2d, overlap_factor=overlap_factor)

    # Create palette of colors for function 'plot_interval_points'
    palette = sns.color_palette("husl", n_colors=num_intervals)

    # For points in each interval compute clusters and store the computed partitions in 'all_partitions'
    # Very important is the selection of eps and min_samples when using DBSCAN.
    interval_pts = []
    all_partitions = []
    cluster_sets = []

    # Axis along which to form intervals
    axis = 0
    for i in range(num_intervals):
        # Compute True/False mask identifying the points that are inside current interval i
        interval_mask = (data2d[:, axis] >= intervals[i][0]) & (data2d[:, axis] < intervals[i][1])
        interval_indices = [index for index, value in enumerate(interval_mask) if value]
        interval_points = data3d[interval_mask]
        clusters = cluster(interval_points, algorithm='AC', dbscan_eps=0.5, dbscan_min_samples=5)
        make_partitions(clusters, interval_indices, all_partitions)

        interval_pts.append(interval_points)
        cluster_sets.append(clusters)

    # Calculate number of colors to be used in the cluster_palette
    # Cluster palette is used in 'plot_interval_points_with_clusters'
    occurrences = [len(np.unique(cls)) for cls in cluster_sets]
    clrs_indices = np.repeat(np.arange(len(occurrences)), occurrences)
    clrs = [palette[i] for i in clrs_indices]
    cluster_palette = sns.color_palette("husl", n_colors=len(clrs))

    # 3d plotting functions with specified order of axes
    plot_interval_points(interval_pts, palette, order=(0, 1, 2))
    plot_interval_points(interval_pts, palette, order=(1, 0, 2))
    plot_interval_points_with_clusters(interval_pts, cluster_sets, cluster_palette, order=(0, 1, 2))
    plot_interval_points_with_clusters(interval_pts, cluster_sets, cluster_palette, order=(1, 0, 2))

    # Create Mapper graph
    # - each cluster is a node
    # - two nodes are connected if their clusters share points
    graph = nx.Graph()
    for i in range(len(all_partitions)):
        graph.add_node(i)
        if i > 0:
            for j in range(i):
                if len(set(all_partitions[i]) & set(all_partitions[j])) > 0:
                    graph.add_edge(i, j)

    # Plot Mapper graph
    plot_mapper(graph, cluster_palette)
    triangles = find_triangles(list(graph.edges))
    # Convert to gudhi simplex tree
    simplex_tree = gudhi.SimplexTree()
    for simplex in graph.edges:
        simplex_tree.insert(simplex)

    simplices = [s[0] for s in list(simplex_tree.get_skeleton(2))]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    draw_2d_simplicial_complex(simplices, ax=ax)
    plt.show()

    # Compute persistent homology
    persistence = simplex_tree.persistence(min_persistence=0.01)
    diagrams = gudhi.plot_persistence_diagram(persistence)
    plt.show()
    # Plot persistent homology diagram
    gudhi.plot_persistence_barcode(persistence)
    plt.show()
