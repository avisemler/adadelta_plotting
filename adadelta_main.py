import math

import numpy as np
import matplotlib.pyplot as plt

DISCRETE_MODE = 1

def loss(u, x_list):
    #exponential loss
    result = 0
    for x in x_list:
        result += math.e ** (- np.dot(u, x))
    return result

def grad_loss(u, x_list):
    #gradient of expontential loss on data points x_list
    result = np.zeros_like(u)
    for x in x_list:
        result += -x * math.e ** (- np.dot(u, x))
    return result


def plot_arrows(subplt, points, label, arrow_freq=5, color="red"):
    arrow_points = points.T[:, ::arrow_freq]
    arrow_pos_x = arrow_points[0, :-1]
    arrow_pos_y = arrow_points[1, :-1]
    arrow_dir_x = arrow_points[0, 1:] - arrow_points[0, :-1]
    arrow_dir_y = arrow_points[1, 1:] - arrow_points[1, :-1]
    subplt.quiver(arrow_pos_x, arrow_pos_y, arrow_dir_x, arrow_dir_y, scale_units="xy", angles="xy", scale=1, color=color, label=label)

def plot_time_curve(ax, points, label, color="red"):
    points = points.T 
    time = np.arange(1, points.shape[1]+1)
    print(points.shape[1], time.shape)
    ax.plot(points[0], points[1], time, color=color, label=label)

#datapoints
x=[np.array([1.5,0]), np.array([0,1.5])]

w = np.array([-1.2, -0.2]) #model
nominator = 0 #conditioner
denominator = 0
epsilon = 0.01
lr = 1 if DISCRETE_MODE else 0.001
rho = 0.9

conditioner = []
grads = []
deltas = []
losses = []
parameter = []

RESOLUTION = int(1/lr)
for i in range(1000 * RESOLUTION):
    grad = grad_loss(w, x)

    denominator = rho * denominator + (1-rho) * grad**2
    nom_scale = 1 if DISCRETE_MODE else 1
    h = np.sqrt(nominator*nom_scale + epsilon) / np.sqrt(denominator + epsilon)
    delta_x = grad * h
    nominator = rho * nominator + (1-rho) * delta_x**2

    w -= lr * delta_x

    conditioner.append(h)
    grads.append(grad)
    deltas.append(delta_x)
    losses.append(loss(w, x))
    parameter.append(w.copy())

    if i % (RESOLUTION*10) == 0:
        #log
        print("Iteration", i, "parameter", w, "conditioner", h, "loss",  loss(w, x))

#result_array = np.array(result)

fig=plt.figure()
#loss=fig.add_subplot(2,1,1)
trajectory=fig.add_subplot(1,1,1)    #,projection='3d')
trajectory.grid(visible=True)
fig.set_figheight(7)
fig.set_figwidth(7)

"""
fig, axs = plt.subplots(2)
loss = axs[0]
trajectory = axs[1]

"""


"""
for dim, subplt in enumerate([indicators_1, indicators_2]):
    subplt.plot([con[dim] for con in conditioner], label='Conditioner h(t)')
    #subplt.plot([delta[dim] for delta in deltas], label='Deltas')
    subplt.plot([grad[dim] for grad in grads], label='Gradient of loss')
    #subplt.plot([p[dim] for p in parameter], label='Parameter value')
    subplt.set_title('Coordinate ' + str(dim))
    subplt.set_xlabel('Time')
    #subplt.set_ylabel('Units')
    subplt.set_ylim([-2, 2])
    subplt.legend()




loss.plot(losses, label="Loss (exponential)")
loss.set_ylim([0.00001, 1000])
loss.set_yscale("log")
loss.set_xlabel("Time")
loss.legend()
loss.set_title("Loss")
"""

#plot trajectory of the parameter w
points = np.array(parameter)
#trajectory.scatter(*points.T, s=1, label="Weights trajectory")

#plot arrows at regular intervals
#plot_arrows(trajectory, points, "Weight trajectory", color="blue")
plot_arrows(trajectory, np.array(conditioner), "h(t) trajectory", color="red")
#plot_arrows(trajectory, np.array(grads), "Gradient trajectory", color="orange")


data = np.array(x)
trajectory.scatter(*data.T, marker="x", label="Data point", color="red")
trajectory.set_title("Trajectories")
trajectory.set_xlabel("Coordinate 0")
trajectory.set_ylabel("Coordinate 1")
trajectory.set_xlim([-0.5, 1.5])
trajectory.set_ylim([-0.5, 1.5])
trajectory.axhline(0,color='black') # x = 0
trajectory.axvline(0,color='black') # y = 0
#trajectory.axline((0, 0), (1, 1), label="Max-margin hyperplane", color="green")
trajectory.legend()

plt.suptitle("Learning rate="+str(lr), fontsize=20)

plt.tight_layout()
plt.savefig("new_adadelta_cond_discrete" + str(DISCRETE_MODE) + ".png", bbox_inches='tight', )
plt.show()
