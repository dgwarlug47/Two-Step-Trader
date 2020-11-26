import torch
import numpy as np
import matplotlib.pyplot as plt
from random import randrange


def get_prepared_data(data_preparation_hyper_parameters):
    # data_preparation_hyper_parameters is simply a way to customize the preparation of data.
    # this function returns in order: [train_data, test_data], real_returns, [train_distribution, test_distribution].
    # Note that the train_data and the test_data have the shape: num_of_securities, num_data_points/seq_len, seq_len, num_features
    # real_returns in the other hand have the shape: num_securities, num_data_points
    # the real_returns comes from the validation_address not the train address
    # the first feature of train_data and test_data is the discretized returns, whereas the real returns only has one feature, which is the real returns
    # the train_distribution and the test_distribution is simply the empirical distribution of the bins in the train_data and validation_test_data
    import time
    time0 = time.clock()

    """ part 1: retreving the information"""
    device = data_preparation_hyper_parameters['device']
    d = data_preparation_hyper_parameters['n_bins']
    batch_size = data_preparation_hyper_parameters['batch_size']
    removing_artificial_data = data_preparation_hyper_parameters['removing_artificial_data']
    train_tickers_names_file = data_preparation_hyper_parameters['train_tickers_names_file_path']
    validation_tickers_names_file = data_preparation_hyper_parameters['validation_tickers_names_file_path']
    train_tickers_base_path = data_preparation_hyper_parameters['train_tickers_base_path']
    validation_tickers_base_path = data_preparation_hyper_parameters['validation_tickers_base_path']

    """ part 2: getting the tickers names from both training and validation addresses """

    train_tickers = loadTickers(train_tickers_names_file)
    validation_tickers = loadTickers(validation_tickers_names_file)

    """ part 3: getting the data from each ticker from both the training and validation addresses """

    all_x, _ = loadData(train_tickers, train_tickers_base_path, data_preparation_hyper_parameters)
    train_data, test_data = train_test_data_creator(all_x, data_preparation_hyper_parameters, False)
    # train_data and test_data have dimensions: (num_securities, num_of_sequences, seq_length, num_of_features)

    validation_all_x, validation_all_real_returns = loadData(validation_tickers, validation_tickers_base_path, data_preparation_hyper_parameters)
    validation_train_data, validation_test_data = train_test_data_creator(validation_all_x, data_preparation_hyper_parameters, True)
    # validation_train_data and validation_test_data have dimensions: (num_securities, num_of_sequences, seq_length, num_of_features)

    # note that if the model set the variable removing_artificial_data to be false
    # the following two lines of code do not remove the artifcial data.
    train_clean = artificial_data_removal(train_data, removing_artificial_data)
    test_clean = artificial_data_removal(test_data, removing_artificial_data)

    """ part 4: bagging the training set """
    train_clean = bootstrap(train_clean, data_preparation_hyper_parameters)

    """ part 5: saving the mean and the variance of the data set features """
    save_mean_and_variance(train_clean, data_preparation_hyper_parameters)

    """ part 6: creating the data loaders for the learning phase """
    train_loader = torch.utils.data.DataLoader(torch.from_numpy(train_clean).to(device), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.from_numpy(test_clean).to(device), batch_size=batch_size, shuffle=True)

    """ part 7: plotting and storing the data distributions """
    store_and_show_true_distribution(data_preparation_hyper_parameters, train_clean, d, title='train_true_distribution')
    store_and_show_true_distribution(data_preparation_hyper_parameters, flatten_data(validation_test_data), d, title='validation_true_distribution')

    time1 = time.clock()
    print("End of the preparation data, ", " this was the total time:", time1 - time0)
    return [torch.FloatTensor(validation_train_data).to(device), torch.FloatTensor(validation_test_data).to(device)], [train_loader, test_loader], validation_all_real_returns


def loadTickers(fileName):
    # receives as input the address where the names of the tickers are stored.
    # outputs the list of the tickers names.
    lineList = list()
    with open(fileName) as f:
        for line in f:
            lineList.append(line)
    lineList = [line.rstrip('\n') for line in open(fileName)]
    return lineList


def loadData(Names_list, base_path, data_preparation_hyper_parameters):
    # Names_list should be the names of the tickers that you are analyzing
    # data_preparation_hyper_parameters is the usual, the base_path with
    # plus the ticker name should give the path to acess the data.
    d = data_preparation_hyper_parameters['n_bins']
    lower_bound = data_preparation_hyper_parameters['lower_bound']
    upper_bound = data_preparation_hyper_parameters['upper_bound']

    x_list = []
    real_returns_list = []
    for name in Names_list:
        path_for_real_returns = base_path + name
        Data = Data = np.genfromtxt(path_for_real_returns)
        # Data should have dimensions (num_data_points, features + 1)
        # Data[:, 0] is the real_returns, where Data[:, 1:] is the features
        x = Data[:, 1:]
        real_returns = Data[:, 0]
        labels = discretization_of_real_returns(real_returns, d, lower_bound, upper_bound)
        # labels has dimensions (num_data_points)
        x = np.concatenate((np.expand_dims(labels, axis=1), x), axis=1)
        x_list.append(torch.from_numpy(x))
        real_returns_list.append(torch.from_numpy(real_returns))
    all_real_returns = torch.cat(real_returns_list).view((len(real_returns_list), real_returns_list[0].shape[0]))
    all_x = torch.cat(x_list).view(len(x_list), x_list[0].shape[0], x_list[0].shape[1])
    return np.array(all_x), np.array(all_real_returns)


def discretization_of_real_returns(real_returns, d, lower_bound, upper_bound):
    # real_returns has dimensions (num_ data_points)
    # the real returns get converted to labels
    bins = np.linspace(lower_bound, upper_bound, d - 1)
    labels = np.digitize(real_returns, bins)
    return labels


def train_test_data_creator(data, data_preparation_hyper_parameters, saving):
    # data should have dimensions (num_securities, num_data_points, num_features)
    # formats the data according to the seq_len, and train_test_validator
    # returns train_data and test_data which should have dimensions
    train_test_division = data_preparation_hyper_parameters['train_test_division']
    seq_length = data_preparation_hyper_parameters['seq_length']
    num_securities = data.shape[0]
    num_data_points = data.shape[1]
    if saving:
        data_preparation_hyper_parameters['remainder_of_test_set'] = num_data_points % seq_length
    n_seqs = int(np.floor(num_data_points / seq_length))
    n1 = int(train_test_division * n_seqs)
    data_shaped = np.reshape(data[:, :seq_length * n_seqs, :], (num_securities, n_seqs, seq_length, -1), order='C')
    train_data = data_shaped[:, :n1, :, :]
    test_data = data_shaped[:, n1:, :]
    return train_data, test_data


def artificial_data_removal(data, removing_artificial_data=False):
    # data has dimensions: num_securties, num_sequences, seq_length, num_features
    # the output has dimension (?,  seq_length, num_features) and it is cleaned (without artificial data).
    flattened_data = flatten_data(data)
    if removing_artificial_data:
        real_data = []
        for index in range(flattened_data.shape[0]):
            difference = (flattened_data[index, :, 1:] != (np.zeros([data.shape[2], data.shape[3] - 1]))).sum()
            if (difference != 0):
                real_data.append(flattened_data[index])
        real_data = np.array(real_data)
        flattened_data = real_data
    return flattened_data


def flatten_data(data):
    # flattens the data
    seq_len = data.shape[2]
    num_features = data.shape[3]
    return np.reshape(data, (-1, seq_len, num_features))


def store_and_show_true_distribution(data_preparation_hyper_parameters, data_clean, num_bins, title):
    # data has dimensions (num_sequences, seq_length, features)
    # num_bins is the number of bins the returns were quantized on.
    # this method simply plots and returns the distribution of the data provided..
    labels = data_clean[:, :, 0]
    total_num_of_data_points = labels.size
    true_distribution = []
    for i in range(num_bins):
        bin_frequency = len(labels[np.where(labels == i)])
        true_distribution.append(bin_frequency / total_num_of_data_points)
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_bins), true_distribution, color='indigo')
    ax.set_title(title)
    plt.show()
    # for index in range(len(true_distribution)):
    #    print('index', index, true_distribution[index])
    data_preparation_hyper_parameters[title] = np.array(true_distribution)


def save_mean_and_variance(flattened_data, data_preparation_hyper_parameters):
    # the dimensions of flattened_data should be  (?, seq_length, num_of_features)
    # note that the we will save the results in the data_preparation.
    # note that the first feature of the flattened_data is the actual label,
    # this will not be normalized. Therefore we do not save the labels average
    # nor the standard deviation.
    num_of_features = flattened_data.shape[2]
    device = data_preparation_hyper_parameters['device']
    thing = np.reshape(flattened_data, (-1, num_of_features))
    data_preparation_hyper_parameters['train_mean'] = torch.FloatTensor(np.mean(thing[:, 1:], axis=0)).to(device)
    data_preparation_hyper_parameters['train_std'] = torch.FloatTensor(np.std(thing[:, 1:], axis=0)).to(device)


def bootstrap(data, data_preparation_hyper_parameters):
    # data has dimensions (num_sequences, seq_length, n_features + 1)
    # data_preparation_hyper_parameters['bootstrap_ratio'] = bagged_data.size[0] / data.size[0]
    # the only exception is when the ratio is equal to 0.
    # in this case there would be no bagging. It simply returns the data
    num_sequences = data.shape[0]
    ratio = data_preparation_hyper_parameters['bootstrap_ratio']
    if ratio == 0:
        return data
    seed = data_preparation_hyper_parameters['bootstrap_seed']

    n_sample = round(num_sequences * ratio)
    bootstrapped_data = list()
    if seed is None:
        while len(bootstrapped_data) < n_sample:
            index = randrange(num_sequences)
            bootstrapped_data.append(data[index])
        return np.array(bootstrapped_data)
    import random
    random.seed(seed)
    bins = np.linspace(0, 1, num_sequences)
    while len(bootstrapped_data) < n_sample:
        key = random.random()
        index = np.digitize([key], bins)[0]
        bootstrapped_data.append(data[index])
    return np.array(bootstrapped_data)
