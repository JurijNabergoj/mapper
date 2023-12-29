import gudhi
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
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
def apply_measurement_function(data, function1="axis0", function2="axis1"):
    functions = {
        "axis0": axis(0),
        "axis1": axis(1),
        "axis2": axis(2),
        "PCA": pca,
        "t-SNE": t_sne,
        "radial": centroid_dist,
    }
    return functions[function1](data), functions[function2](data)


# partitions data into rectangles based on two measurement functions
def partition(data, f_values, g_values, num_rectangles=None, overlap=0.2):
    num_rectangles = int(np.sqrt(data.shape[0])) if num_rectangles is None else num_rectangles

    min_f, max_f = np.min(f_values), np.max(f_values)
    min_g, max_g = np.min(g_values), np.max(g_values)

    overlap_size_f = overlap * (max_f - min_f) / num_rectangles
    overlap_size_g = overlap * (max_g - min_g) / num_rectangles

    rectangles = []
    rectangle_indices = []
    for i in range(num_rectangles):
        for j in range(num_rectangles):
            min_f_range = min_f + i * (max_f - min_f) / num_rectangles
            max_f_range = min_f + (i + 1) * (max_f - min_f) / num_rectangles
            min_f_range -= overlap_size_f
            max_f_range += overlap_size_f

            min_g_range = min_g + j * (max_g - min_g) / num_rectangles
            max_g_range = min_g + (j + 1) * (max_g - min_g) / num_rectangles
            min_g_range -= overlap_size_g
            max_g_range += overlap_size_g

            selected_points = []
            selected_indices = []
            for index, (point, f_value, g_value) in enumerate(zip(data, f_values, g_values)):
                if (min_f_range < f_value < max_f_range) and (min_g_range < g_value < max_g_range):
                    selected_points.append(point)
                    selected_indices.append(index)

            rectangles.append(np.array(selected_points))
            rectangle_indices.append(np.array(selected_indices))

    return rectangles, rectangle_indices


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
        data, measurement_function1="axis0", measurement_function2="axis2", n_partitions=None, overlap=0.2, plot=False
):
    measure1, measure2 = apply_measurement_function(data, function1=measurement_function1,
                                                    function2=measurement_function2)
    partitions, indices = partition(data, measure1, measure2, num_rectangles=n_partitions, overlap=overlap)
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


def mapper_algorithm(data, num_rectangles=5, overlap=0.15, algorithm="AC", f="axis0", g="axis2"):
    rectangles, rectangle_indices = partition_data(
        data,
        measurement_function1=f,
        measurement_function2=g,
        n_partitions=num_rectangles,
        overlap=overlap,
        plot=True
    )
    vertices = construct_vertices(data, rectangles, rectangle_indices, algorithm=algorithm)
    edges, edge_mappings = construct_edges(vertices)
    triangles, triangle_mappings = construct_triangles(vertices)
    tetrahedra, tetrahedra_mappings = construct_tetrahedra(vertices)

    return vertices, edges, triangles, tetrahedra, edge_mappings, triangle_mappings, tetrahedra_mappings


def construct_vertices(data, rectangles, rectangle_indices, algorithm='AC'):
    vertices = {}

    for i in range(len(rectangles)):
        if len(rectangles[i]) == 0:
            continue
        selected_data = data[rectangle_indices[i]]
        labels = cluster(selected_data, algorithm=algorithm)
        unique_labels = np.unique(labels)
        cluster_centers = [np.mean(selected_data[labels == label], axis=0) for label in unique_labels]

        for label, center in zip(unique_labels, cluster_centers):
            vertices[(i, label)] = {'center': center, 'data_indices': rectangle_indices[i][labels == label]}

    return vertices


def construct_edges(vertices):
    vertices_keys = list(vertices.keys())
    edges = []
    edge_mappings = []
    for (i, j) in vertices_keys:
        index_i_j = vertices_keys.index((i, j))
        for (k, l) in vertices_keys[index_i_j + 1:]:
            if (i, j) != (k, l):
                if len(np.intersect1d(vertices[(i, j)]['data_indices'], vertices[(k, l)]['data_indices'])) > 0:
                    edges.append((vertices[(i, j)]['center'], vertices[(k, l)]['center']))
                    edge_mappings.append([(i, j)] + [(k, l)])
    return edges, edge_mappings


def construct_triangles(vertices):
    vertices_keys = list(vertices.keys())
    triangles = []
    triangle_mappings = []
    for (i, j) in vertices_keys:
        index_i_j = vertices_keys.index((i, j))
        for (k, l) in vertices_keys[index_i_j:]:
            index_k_l = vertices_keys.index((k, l))
            if (i, j) != (k, l):
                if len(np.intersect1d(vertices[(i, j)]['data_indices'], vertices[(k, l)]['data_indices'])) > 0:
                    for (m, n) in vertices_keys[index_k_l + 1:]:
                        if (i, j) != (m, n) and (k, l) != (m, n):
                            if (len(np.intersect1d(vertices[(i, j)]['data_indices'],
                                                   vertices[(m, n)]['data_indices'])) > 0 and
                                    len(np.intersect1d(vertices[(k, l)]['data_indices'],
                                                       vertices[(m, n)]['data_indices'])) > 0):
                                triangles.append((
                                    vertices[(i, j)]['center'],
                                    vertices[(k, l)]['center'],
                                    vertices[(m, n)]['center']
                                ))
                                triangle_mappings.append([(i, j)] + [(k, l)] + [(m, n)])
    return triangles, triangle_mappings


def construct_tetrahedra(vertices):
    vertices_keys = list(vertices.keys())
    tetrahedra = []
    tetrahedra_mappings = []
    for (i, j) in vertices_keys:
        index_i_j = vertices_keys.index((i, j))

        for (k, l) in vertices_keys[index_i_j:]:
            index_k_l = vertices_keys.index((k, l))
            if (i, j) != (k, l):
                if len(np.intersect1d(vertices[(i, j)]['data_indices'], vertices[(k, l)]['data_indices'])) > 0:
                    for (m, n) in vertices_keys[index_k_l + 1:]:
                        index_m_n = vertices_keys.index((m, n))
                        if (i, j) != (m, n) and (k, l) != (m, n):
                            if (len(np.intersect1d(vertices[(i, j)]['data_indices'],
                                                   vertices[(m, n)]['data_indices'])) > 0 and
                                    len(np.intersect1d(vertices[(k, l)]['data_indices'],
                                                       vertices[(m, n)]['data_indices'])) > 0):
                                for (o, p) in vertices_keys[index_m_n + 1:]:
                                    if (i, j) != (o, p) and (k, l) != (o, p) and (m, n) != (o, p):
                                        if (len(np.intersect1d(vertices[(i, j)]['data_indices'],
                                                               vertices[(o, p)]['data_indices'])) > 0 and
                                                len(np.intersect1d(vertices[(k, l)]['data_indices'],
                                                                   vertices[(o, p)]['data_indices'])) > 0 and
                                                len(np.intersect1d(vertices[(m, n)]['data_indices'],
                                                                   vertices[(o, p)]['data_indices'])) > 0):
                                            tetrahedra.append((
                                                vertices[(i, j)]['center'],
                                                vertices[(k, l)]['center'],
                                                vertices[(m, n)]['center'],
                                                vertices[(o, p)]['center']
                                            ))
                                            tetrahedra_mappings.append([(i, j)] + [(k, l)] + [(m, n)] + [(o, p)])

    return tetrahedra, tetrahedra_mappings


def plot_result(vertices, edges, triangles, tetrahedra):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for vertex in vertices.values():
        ax.scatter(vertex['center'][0], vertex['center'][1], 0, c='r', marker='o')

    for edge in edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [0, 0], c='b')

    for triangle in triangles:
        triangle = np.concatenate((triangle, [triangle[0]]))  # Close the triangle
        ax.plot(triangle[:, 0], triangle[:, 1], [0, 0, 0, 0], c='g')

    for tetrahedron in tetrahedra:
        tetrahedron = np.concatenate((tetrahedron, [tetrahedron[0]]))  # Close the tetrahedron
        ax.plot(tetrahedron[:, 0], tetrahedron[:, 1], [0, 0, 0, 0, 0], c='m')

    ax.set_title('Mapper Simplicial complex')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.tight_layout()
    plt.show()


def map_simplices(vertices, edge_mappings, triangle_mappings, tetrahedra_mappings):
    vertex_to_ind = {vertex: i for i, vertex in enumerate(vertices)}

    mapped_vertices = [vertex_to_ind[v] for v in list(vertices.keys())]
    mapped_edges = [(vertex_to_ind[e1], vertex_to_ind[e2]) for (e1, e2) in edge_mappings]
    mapped_triangles = [(vertex_to_ind[e1], vertex_to_ind[e2], vertex_to_ind[e3]) for (e1, e2, e3) in triangle_mappings]
    mapped_tetrahedra = [(vertex_to_ind[e1], vertex_to_ind[e2], vertex_to_ind[e3], vertex_to_ind[e4]) for
                         (e1, e2, e3, e4) in tetrahedra_mappings]
    return mapped_vertices, mapped_edges, mapped_triangles, mapped_tetrahedra


if __name__ == "__main__":
    table = read3d("data/table.ply", density=0.2, resize=0.15)
    data = join_shapes(k_torus(1), table, np.array([0, -10, 0]), np.array([0, 10, 0]))
    plot3d(data)
    vertices, edges, triangles, tetrahedra, edge_mappings, triangle_mappings, tetrahedra_mappings = mapper_algorithm(
        data, num_rectangles=10, algorithm="AC", f="axis0", g="axis1")
    plot_result(vertices, edges, triangles, tetrahedra)
    mapped_vertices, mapped_edges, mapped_triangles, mapped_tetrahedra = map_simplices(vertices, edge_mappings,
                                                                                       triangle_mappings,
                                                                                       tetrahedra_mappings)
    # Construct the simplex tree
    # Tukej nevem tocno kaj nej bi bil filtration value za vsak simplex zato sm dal: dim + random value.
    # Kaj bi ti dal?
    simplex_tree = gudhi.SimplexTree()
    for simplex in mapped_edges:
        simplex_tree.insert(simplex, filtration=np.random.rand())
    for simplex in mapped_triangles:
        simplex_tree.insert(simplex, filtration=1 + np.random.rand())
    for simplex in mapped_tetrahedra:
        simplex_tree.insert(simplex, filtration=2 + np.random.rand())

    # Compute persistent homology
    persistence = simplex_tree.persistence(min_persistence=-1.0, persistence_dim_max=True)
    diagrams = gudhi.plot_persistence_diagram(persistence)
    plt.tight_layout()
    plt.show()

    # Plot persistent homology diagram
    gudhi.plot_persistence_barcode(persistence)
    plt.tight_layout()
    plt.show()
