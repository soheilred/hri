import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 4,3
from matplotlib.animation import FuncAnimation


file_name = '/home/soheil/Sync/unh/courses/hri/project/extern/exercise_data/data/Subject 11/subject_11_3R0001.tsv'
# points_df = pd.read_csv(file_name, delimiter="\t")

names = []
data_list = []
with open(file_name) as f:
    for i, line in enumerate(f):
        if i == 9:
            line = line.rstrip('\n\r')
            l = line.split('\t')
            names.extend(l)
        if i > 9:
            line = line.rstrip('\n\r')
            # l = line.split('\t')
            l = line.replace('\t', ' ')
            # print(l)
            nums = l.split(' ')
            data_list.append(nums)
            # print(nums)
data = np.array(data_list, dtype=float)

# create a figure with an axes
fig, ax = plt.subplots()

# set the axes limits
ax.axis([0, 1500, 0, 1500])
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

# set equal aspect such that the circle is not shown as ellipse
ax.set_aspect("equal")

# create a point in the axes
point, = ax.plot(0,1, marker="o")

# Updating function, to be repeatedly called by the animation
def update(i):
    # obtain point coordinates 
    x = data[i, 0::3]
    y = data[i, 1::3]
    # import ipdb; ipdb.set_trace()
    # set point's coordinates
    point.set_data([x], [y])
    return point,

# create animation with 10ms interval, which is repeated,
# provide the full circle (0,2pi) as parameters
ani = FuncAnimation(fig, update, interval=data.shape[0] // 250, blit=True, repeat=True,
                    frames=data.shape[0])
                    # frames=np.linspace(0,2*np.pi,360, endpoint=False))

# ani = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)

ani.save('animation.mp4', fps=250, extra_args=['-vcodec', 'libx264'])
plt.show()
print(data.shape)
