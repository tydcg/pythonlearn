# 学习动画
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from mpl_toolkits.mplot3d import Axes3D, art3d


# 制作一个动画效果的3d图
# 先平铺好x,y轴，然后随机生成v
# 然后画出图像，之后再不停的去修改这个图像

class AnimatoinExample:

    # 先平铺好x,y轴，然后随机生成v
    # 然后画出图像
    def __init__(self):
        self.xp, self.yp = np.meshgrid(np.arange(1, 10, 1), np.arange(1, 10, 1))
        # 随机生成一个9*9的二维矩阵，矩阵的值的范围是[1-10)
        self.z = np.random.uniform(1, 10, (9, 9))

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")

        # 设置x,y,z轴的最大显示
        self.ax.set_xlim([0, 11])
        self.ax.set_ylim([0, 11])
        self.ax.set_zlim([0, 11])
        # 关掉坐标轴的显示
        plt.axis('off')

        # 画图，使用rainbow彩虹色
        self.pc = self.ax.plot_surface(self.xp, self.yp, self.z, cmap='rainbow')

        # 画投影
        self.ax.contour(self.xp, self.yp, self.z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))

    def array_perimeter(self, arr):
        forward = np.s_[0:-1]  # [0 ... -1)
        backward = np.s_[-1:0:-1]  # [-1 ... 0)
        return np.concatenate((
            arr[0, forward],
            arr[forward, -1],
            arr[-1, backward],
            arr[backward, 0],
        ))

    # 将二维矩阵z转化成多个多边形的点
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
                ps = np.array(ps).T
                polys.append(ps)
        return polys

    # 更新poly3dCollection的z值
    def update(self, data):
        # self.ax.contour(self.xp, self.yp, self.z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
        self.pc.set(verts=self.to3d(data), cmap='rainbow')
        # return p

    # 随机生成一个9*9的二维矩阵，矩阵的值的范围是[1-10)
    def generated(self):
        while True:
            z = np.random.uniform(1, 10, (9, 9))
            yield z

    def start(self):
        # 这里的anim必须有一个地方持有它的引用，不然它会被GC掉
        self.anim = animation.FuncAnimation(self.fig, self.update, self.generated, interval=150)
        plt.show()
        # self.fig.savefig("surface3d_frontpage.png", dpi=600) # 保存图


example = AnimatoinExample()
example.start()
