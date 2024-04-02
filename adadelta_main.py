import math

import numpy as np
import matplotlib.pyplot as plt

#Set to 1 to run with lr=1, or 0 to run with small lr
DISCRETE_MODE = 0

def loss(u, x_list):
    """Exponential loss of linear classification vector u on dataset x_list"""
    result = 0
    for x in x_list:
        result += math.e ** (- np.dot(u, x))
    return result

def grad_loss(u, x_list):
    """Gradient of expontential loss on data points x_list"""
    result = np.zeros_like(u)
    for x in x_list:
        #sum dataset gradients
        result += -x * math.e ** (- np.dot(u, x))
    return result


def plot_arrows(subplt, points, label, arrow_freq=5, color="red"):
    """Plot points with arrows joining"""
    arrow_points = points.T[:, ::arrow_freq]
    arrow_pos_x = arrow_points[0, :-1]
    arrow_pos_y = arrow_points[1, :-1]
    arrow_dir_x = arrow_points[0, 1:] - arrow_points[0, :-1]
    arrow_dir_y = arrow_points[1, 1:] - arrow_points[1, :-1]
    subplt.quiver(arrow_pos_x, arrow_pos_y, arrow_dir_x, arrow_dir_y, scale_units="xy", angles="xy", scale=1, color=color, label=label)

def plot_time_curve(ax, points, label, color="red"):
    points = points.T 
    time = np.arange(1, points.shape[1]+1)
    ax.plot(points[0], points[1], time, color=color, label=label)

#datapoints for training
x=[np.array([1.5,0]), np.array([0,1.5])]

w = np.array([-1.2, -0.2]) #iniitalisation of linear classifier

#moving averages for conditioner
numerator = 0 
denominator = 0

epsilon = 0.01
lr = 1 if DISCRETE_MODE else 0.001
rho = 0.9

#store the values over the training
conditioner = []
grads = []
deltas = []
losses = []
parameter = []

#e.g., if the lr is 10x smaller, run for 10x longer
resolution = int(1/lr)
for i in range(1000 * resolution):
    grad = grad_loss(w, x)

    denominator = rho * denominator + (1-rho) * grad**2
    h = np.sqrt(numerator + epsilon) / np.sqrt(denominator + epsilon)
    delta_x = grad * h
    numerator = rho * numerator + (1-rho) * delta_x**2

    #update parameter
    w -= lr * delta_x

    conditioner.append(h)
    grads.append(grad)
    deltas.append(delta_x)
    losses.append(loss(w, x))
    parameter.append(w.copy())

    if i % (resolution*10) == 0:
        #log
        print("Iteration", i, "parameter", w, "conditioner", h, "loss",  loss(w, x))

fig=plt.figure()
#loss=fig.add_subplot(2,1,1)
trajectory=fig.add_subplot(1,1,1)    #,projection='3d')
trajectory.grid(visible=True)
fig.set_figheight(7)
fig.set_figwidth(7)

#plot trajectory of the parameter w
plot_arrows(trajectory, np.array(parameter), "Weight trajectory", color="blue")
plot_arrows(trajectory, np.array(conditioner), "h(t) trajectory", color="red")
#plot_arrows(trajectory, np.array(grads), "Gradient trajectory", color="orange")

#plot the data
data = np.array(x)
trajectory.scatter(*data.T, marker="x", label="Data point", color="red")

trajectory.set_title("Trajectories")
trajectory.set_xlabel("Coordinate 0")
trajectory.set_ylabel("Coordinate 1")
trajectory.set_xlim([-2, 7])
trajectory.set_ylim([-2, 7])
trajectory.axhline(0,color='black') # x = 0
trajectory.axvline(0,color='black') # y = 0
#trajectory.axline((0, 0), (1, 1), label="Max-margin hyperplane", color="green")
trajectory.legend()

plt.suptitle("Learning rate="+str(lr), fontsize=20)

plt.tight_layout()
plt.savefig("output_discrete" + str(DISCRETE_MODE) + ".png", bbox_inches='tight', )
plt.show()
