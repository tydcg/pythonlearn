# 语法测试
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, art3d

x = np.arange(1, 4, 1)
y = np.arange(1, 4, 1)
z = np.arange(1, 4, 1)

x, y = np.meshgrid(x, y)
z, _ = np.meshgrid(z, z)


def to3D(z):
    A = x.flatten()[:, np.newaxis]
    B = y.flatten()[:, np.newaxis]
    C = z.flatten()[:, np.newaxis]

    D = np.hstack((A, B))  # horizontal stack
    D = np.hstack((D, C))
    return D


def array_perimeter(arr):
    forward = np.s_[0:-1]  # [0 ... -1)
    backward = np.s_[-1:0:-1]  # [-1 ... 0)
    return np.concatenate((
        arr[0, forward],
        arr[forward, -1],
        arr[-1, backward],
        arr[backward, 0],
    ))


x = np.arange(1, 10, 1)
y = np.arange(1, 10, 1)
z = np.arange(1, 10, 1)

xp, yp = np.meshgrid(x, y)
# z, _ = np.meshgrid(z, y)
z = np.random.uniform(1, 10, (9, 9))

rows, cols = z.shape

rstride = 2
cstride = 2

row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]

polys = []
for rs, rs_next in zip(row_inds[:-1], row_inds[1:]):
    for cs, cs_next in zip(col_inds[:-1], col_inds[1:]):
        ps = [
            # +1 ensures we share edges between polygons
            array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1])
            for a in (xp, yp, z)
        ]
        # ps = np.stack(ps, axis=-1)
        ps = np.array(ps).T
        polys.append(ps)

print(polys)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

ax.set_xlim([0, 11])
ax.set_ylim([0, 11])
ax.set_zlim([0, 11])

pc = art3d.Poly3DCollection(polys, cmap='rainbow')
ax.add_collection(pc)

print(polys)

plt.show()

# exit()
