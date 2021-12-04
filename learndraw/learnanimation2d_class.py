# 学习动画
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 2d 动画图
class Animatoin2dExample:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.N = 10
        self.x = np.random.rand(self.N)
        self.y = np.random.rand(self.N)
        self.z = np.random.rand(self.N)

        self.circles, self.triangles, self.dots = self.ax.plot(self.x, 'ro', self.y, 'g^', self.z, 'b')

        self.ax.set_ylim(0, 1)
        # plt.axis('off')

    def update(self, data):
        # print(data)
        self.circles.set_ydata(data[0])
        self.triangles.set_ydata(data[1])
        self.dots.set_ydata(data[2])
        return self.circles, self.triangles, self.dots

    def generated(self):
        while True:
            yield np.random.rand(3, self.N)

    def start(self):
        # 这里的anim必须有一个地方持有它的引用，不然它会被GC掉
        self.anim = animation.FuncAnimation(self.fig, self.update, self.generated, interval=150)
        plt.show()


example = Animatoin2dExample()
example.start()
