import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.animation import FuncAnimation


def init_faces(N):
    f = []
    for r in range(N - 1):
        for c in range(N - 1):
            v0 = r * N + c
            f.append([v0, v0 + 1, v0 + N + 1, v0 + N])
    return np.array(f)


def init_vert(N):
    v = np.meshgrid(range(N), range(N), [1.0])
    return np.dstack(v).reshape(-1, 3)


def set_amplitude(v, A):
    v[:, 2] = A * (np.sin(np.pi * v[:, 0] / (N - 1)) * np.sin(np.pi * v[:, 1] / (N - 1)))
    return v


N = 10
f = init_faces(N)
v = init_vert(N)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")

pc = art3d.Poly3DCollection(v[f])
# pc.set_animated(True)  # Is this required? Why?

ax.add_collection(pc)

def init_fig():
    ax.set_xlim([0, N])
    ax.set_ylim([0, N])
    ax.set_zlim([0, 5])
    return pc,


def update_fig(frame):
    A = np.sin(frame)
    new_v = set_amplitude(v, A)
    pc.set_verts(new_v[f])
    return pc

def gen():
    while True:
        yield np.linspace(0, 2 * np.pi, 128)

ani = FuncAnimation(fig, update_fig, frames=np.linspace(0, 2 * np.pi, 128),
                    init_func=init_fig, blit=False, repeat=True)

# ani = FuncAnimation(fig, update_fig, gen,
#                     init_func=init_fig, blit=False, interval=150)

plt.show()
