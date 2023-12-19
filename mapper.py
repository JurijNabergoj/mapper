from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


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


# =====================================================================


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
ax.set_title("3D plot")
plt.show()

# get 2d data
data2d = filter(data3d, lens="t-SNE")

# plot 2d
data2d = data2d.T
plt.scatter(data2d[0], data2d[1])
plt.show()

print(cluster(data2d))
