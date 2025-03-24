import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def read_data(path):

    df = pd.read_csv(path)
    data_np = df.to_numpy()

    return data_np

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default="fixed")
args = vars(parser.parse_args())


grid_side = 50

lat = np.linspace(0, 1, num=grid_side)
long = np.linspace(0, 1, num=grid_side)

x_mesh, y_mesh = np.meshgrid(lat, long)
x_unraveled = np.reshape(x_mesh, (-1,1))
y_unraveled = np.reshape(y_mesh, (-1,1))
location = np.array((x_unraveled, y_unraveled))
location = np.squeeze(location)
location = location.T
#print(location.shape) # (2500, 2)

x = np.reshape(x_unraveled, (grid_side, grid_side))
y = np.reshape(y_unraveled, (grid_side, grid_side))

if args['dataset'] == "fixed":
    path = "../datasets/synthetic/fixed_test.csv"

else:
    path = "../datasets/synthetic/flight_test.csv"

data_np = read_data(path) 
# data shape: (1393, 4)
# columns: (time, x-coord, y-coord, z-value)

plt.figure()

pz = np.array((data_np[:,0]-np.min(data_np[:,0]))/(np.max(data_np[:,0])-np.min(data_np[:,0])))
colors = np.array([[176/255, 10/255, 20/255, 1.]]*(len(data_np)))

colors[:,-1] = pz
plt.scatter(data_np[:,1], data_np[:,2], color=colors, marker='o')
plt.show()

