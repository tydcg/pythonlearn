# 语法测试
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, art3d


class test:

    def __init__(self):
        self.pc = None
        self.xp, self.yp = np.meshgrid(np.arange(1, 10, 1), np.arange(1, 10, 1))
        self.z = np.random.uniform(1, 10, (9, 9))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

    def array_perimeter(self, arr):
        forward = np.s_[0:-1]  # [0 ... -1)
        backward = np.s_[-1:0:-1]  # [-1 ... 0)
        return np.concatenate((
            arr[0, forward],
            arr[forward, -1],
            arr[-1, backward],
            arr[backward, 0],
        ))

    def to3d(self, z):
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
                    self.array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1])
                    for a in (self.xp, self.yp, z)
                ]
                # ps = np.stack(ps, axis=-1)
                ps = np.array(ps).T
                polys.append(ps)
        return polys

    def update(self, data):
        self.pc.set_verts(self.to3d(data))
        self.pc.set(cmap='rainbow')
        return self.pc

    def generated(self):
        while True:
            z = np.random.uniform(1, 10, (9, 9))
            yield z

    def draw(self):

        self.ax.set_xlim([0, 11])
        self.ax.set_ylim([0, 11])
        self.ax.set_zlim([0, 11])

        # self.pc = art3d.Poly3DCollection(self.to3d(self.z))
        self.pc = self.ax.plot_surface(self.xp, self.yp, self.z, cmap='rainbow')
        self.ax.contour(self.xp, self.yp, self.z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
        self.ax.add_collection(self.pc)

        self.anim = animation.FuncAnimation(self.fig, self.update, self.generated, interval=150)
        plt.show()


# exit()

t = test()
t.draw()
