import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


def drawLine():
    x = np.linspace(-1, 1, 50)
    y = 2 * x + 1

    plt.figure()
    plt.plot(x, y)


def draw_scatter():
    n = 1024
    X = np.random.normal(0, 1, 1024)  # 正态分布
    Y = np.random.normal(0, 1, 1024)
    T = np.arctan2(X, Y)  # for color

    plt.figure()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks(())
    plt.yticks(())

    plt.scatter(X, Y, c=T, alpha=0.65)


def draw_image():
    img = Image.new('RGB', (255, 255), "black")

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            img.putpixel((i, j), (i, j, i));

    img.show()


def draw_image3D():
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    # 定义三维数据
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(X) + np.cos(Y)

    # 作图
    # ax3.plot_surface(X, Y, Z, cmap='rainbow')
    ax3.plot_surface(X, Y, Z, c='r')



def draw_image3DMy():
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    # 定义三维数据
    x = np.linspace(1, 10, 10)
    y = np.linspace(1, 10, 10)

    z = []

    for i in range(x.size):
        z.append(x[i] + y[i])

    z = np.array(z)
    z, _ = np.meshgrid(z, y)
    x, y = np.meshgrid(x, y)

    # 作图
    ax3.plot_surface(x, y, z, cmap='rainbow')


def draw_image3DMy():
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    # 定义三维数据
    x = np.linspace(1, 10, 10)
    y = np.linspace(1, 10, 10)

    z = []
    m = []

    for i in range(x.size):
        z.append(x[i] * x[i] * y[i] * y[i])

    z = np.array(z)
    z, _ = np.meshgrid(z, y)
    x, y = np.meshgrid(x, y)

    # 作图
    ax3.plot_surface(x, y, z, cmap='rainbow')
    # ax3.scatter(x, y, z, c='r')

# drawLine()
# draw_scatter()
# plt.show()

# draw_image()
# draw_image3D()

draw_image3DMy()

plt.show()
