import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

import argparse

def plot_map(x, y, true, temp, filename, confidence=False):

    max_limit = np.max(true)
    min_limit = np.min(true)
    
    temp = np.reshape(temp, (grid_side, grid_side))
    
    fig, ax = plt.subplots(1,1)

    if confidence is False:
        cp = ax.contourf(x, y, temp.T, cmap='coolwarm', vmin=min_limit, vmax=max_limit)
    else:
        cp = ax.contourf(x, y, temp.T, cmap='coolwarm')
    fig.colorbar(cp) # Add a colorbar to a plot

    ax.set_title('Temperature Map')
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')

    plt.savefig(filename)
    plt.close()

def plot_simple(temp, true, filename):

    max_limit = np.max(true)
    min_limit = np.min(true)

    plt.figure()
    plt.imshow(temp, vmin=min_limit, vmax=max_limit, cmap='RdYlGn_r') # (1, 50, 50, 1)
    plt.colorbar()

    plt.savefig(filename)
    
    plt.close()
    #plt.show()

def predict(features_to_predict, train_features, train_labels):

    temp_list = [] 

    for i in range(n_samples):
        # sample a subset of the data points for bootstrapping
        selection = np.random.choice(np.arange(0,len(train_features)), size=int(len(train_features)*prob), replace=True)

        train_features_subset = train_features[selection] # shape: (len(data_from_db)*prob, 3)
        train_labels_subset = train_labels[selection]

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(train_features_subset, train_labels_subset)

        #if i%10==0:
        #    print("[INFO] {} samples processed".format(i))

        # inference
        temp_list.append(model.predict(features_to_predict))

    temp_array = np.asarray(temp_list) # shape: (n_samples, size_of_mesh*size_of_mesh)

    temp_mean = np.mean(temp_array, axis=0) # shape: (size_of_mesh*size_of_mesh,)
    temp_std = np.std(temp_array, axis=0) # shape: (size_of_mesh*size_of_mesh,)

    return temp_mean, temp_std

def read_data(path):

    df = pd.read_csv(path)
    data_np = df.to_numpy()

    print(f'data shape: {data_np.shape}')

    return data_np

def normalize(data, by=2*np.pi):
    data[:,0] /= by
    return data


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default="fixed")
parser.add_argument('-d', '--dataset', default="synthetic")
args = vars(parser.parse_args())


# hyperparameters: 
# k neighbors, 
# prob, 
# normalizing factor (larger factor, more importance to time => shortens time dimension)

if args['dataset'] == 'synthetic':
    
    print("Synthetic dataset")
    
    grid_side = 50

    lat = np.linspace(0, 1, num=grid_side)
    long = np.linspace(0, 1, num=grid_side)

    x_mesh, y_mesh = np.meshgrid(lat, long)
    x_unraveled = np.reshape(x_mesh, (-1,1))
    y_unraveled = np.reshape(y_mesh, (-1,1))
    location = np.array((x_unraveled, y_unraveled))
    location = np.squeeze(location)
    location = location.T

    x = np.reshape(x_unraveled, (grid_side, grid_side))
    y = np.reshape(y_unraveled, (grid_side, grid_side))

    if args['type'] == "fixed":
        print("fixed sensors")

        path = "../datasets/synthetic/fixed_test.csv"
        

        # knn variables
        # fixed
        n_samples = 160
        prob = 0.6
        n_neighbors = 3
        normalize_by = 1.0

        data_np = read_data(path)
        max_time = np.max(data_np[:,0])

    else:
        print("flight sensors")

        path = "../datasets/synthetic/flight_test.csv"

        # flight
        n_samples = 160 #90
        prob = 0.4
        n_neighbors = 3
        normalize_by = 2.174

        data_np = read_data(path)
        max_time = 24.10

    y_true = np.sin(max_time) * location[:,0] + np.cos(max_time) * location[:,1]

else: 

    print("TaxiBJ dataset")

    # hyperparameters: 
    # k neighbors, 
    # prob, 
    # normalizing factor (larger factor, more importance to time => shorten time dimension)
    if args['type'] == "fixed":
        print("fixed sensors")

        path = "../datasets/taxibj/fixed_test_50.npy"

        # knn variables
        # fixed
        n_samples = 160
        prob = 0.9
        n_neighbors = 3
        normalize_by = 48 

    else:
        print("flight sensors")

        path = "../datasets/taxibj/flight_test_50.npy"

        # flight
        n_samples = 160 #90
        prob = 0.7
        n_neighbors = 4
        normalize_by = 144

    data_np = data_np = np.load(path)

    max_time = np.ceil(np.max(data_np[:,0])) # necessary np.ceil for flight data, because time is a float point and not integer

    data1000 = np.load("../datasets/taxibj/data_50_0_1000.npy")
    data2000 = np.load('../datasets/taxibj/data_50_1000_2000.npy')
    data3000 = np.load('../datasets/taxibj/data_50_2000_3000.npy')
    data4000 = np.load('../datasets/taxibj/data_50_3000_4000.npy')
    data5000 = np.load('../datasets/taxibj/data_50_4000_5000.npy')
    data5664 = np.load('../datasets/taxibj/data_50_5000_5664.npy')

    ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

    data_true = ground_truth[np.where(ground_truth[:,0] == max_time)]
    y_true = data_true[:,3]

    location = data_true[:,1:3]

mean_squared_error_list = []

data_normalized = normalize(data_np, by=normalize_by)

if args['dataset'] != 'synthetic':
    data_normalized[:,1] = data_normalized[:,1] / 49.0
    data_normalized[:,2] = data_normalized[:,2] / 49.0

train_features = data_normalized[:,:3] # (100, 3)
train_labels = data_normalized[:, -1] # (100,)

t_sample = np.reshape(np.array([max_time/normalize_by]*2500), (-1,1))

features_to_predict = np.concatenate([t_sample, location], axis=1)

temp_mean, temp_confidence = predict(features_to_predict, train_features, train_labels)
    
mean_squared_error = (np.mean(np.square(temp_mean - y_true)))

if args['dataset'] == 'synthetic':
    plot_map(x, y, y_true, y_true, f'plots/true_{args['dataset']}_{args['type']}.png')
    plot_map(x, y, y_true, temp_mean, f'plots/predicted_{args['dataset']}_{args['type']}.png')
else:
    plot_simple(np.reshape(y_true, (50,50)), y_true,  f'plots/true_{args['dataset']}_{args['type']}.png')
    plot_simple(np.reshape(temp_mean, (50,50)), y_true, f'plots/predicted_{args['dataset']}_{args['type']}.png')

print(f'MSE: {(mean_squared_error)}')
print(f'RMSE: {np.sqrt(mean_squared_error)}')

