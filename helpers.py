
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from os.path import exists, dirname
import pickle


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def save_training_plot(train_losses, test_losses, title, fname, yticks=True):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    if (yticks):
        plt.yticks(np.arange(0, max(test_losses), 0.2))
    # savefig(fname)
    plt.show()


def list_iterator(lists):
    if len(lists) == 1:
        return map(lambda x: [x], lists[0])
    else:
        all_things = []
        things = list_iterator(lists[1:])
        for thing in things:
            for new_guy in lists[0]:
                all_things.append([new_guy] + thing)
        return all_things


def hyper_parameters_iterator(lists):
    if len(lists.keys()) == 1:
        name = list(lists.keys())[0]
        return [{name: ll} for ll in lists[name]]
    else:
        all_things = []
        all_but_first = lists.copy()
        first_name = list(lists.keys())[0]
        del all_but_first[first_name]
        things = hyper_parameters_iterator(all_but_first)
        for thing in things:
            for new_guy in lists[first_name]:
                new_thing = thing.copy()
                new_thing[first_name] = new_guy
                all_things.append(new_thing)
        return all_things


def to_one_hot(labels, d, seq_len, device=torch.device('cuda')):
    print('labels', labels.shape)
    print('seq_len', seq_len)
    # one_hot = torch.FloatTensor(labels.shape[0], d).cuda()
    one_hot = torch.FloatTensor(labels.shape[0], seq_len, d).to(device)
    one_hot.zero_()
    one_hot.scatter_(2, labels.unsqueeze(-1).to(device), 1)
    return one_hot

# This method, is a quantizer, that is it helps to turn discrete
# inputs into continuous ones.


def sample_returns(sample_np, bins, using_middle_returns):
    # sample_np is a numpy array of labels.
    # bins is an array where, bins[i] is the return of the
    # ith label.
    rets = []
    for i in range(len(sample_np)):
        if sample_np[i] == 0:
            ret = bins[0]
        elif sample_np[i] == len(bins):
            ret = bins[len(bins) - 1]
        else:
            lim_inf = bins[sample_np[i] - 1]
            lim_sup = bins[sample_np[i]]
            if using_middle_returns:
                std = lim_inf + (lim_sup - lim_inf) * 0.5
            else:
                std = lim_inf + (lim_sup - lim_inf) * np.random.random()
            ret = std
        rets.append(ret)
    return np.array(rets)


def add_info(cums, index, hyper_parameters, trial_summary):
    # trial_summary is the trial summary of the trial with the hyper_parameters=hyper_parameters
    # this code basically adds this information to the cums[index], by basically  doing:
    # cums[index][hyper_parameters] = trial_summary
    hyper_parameters = tuple(hyper_parameters)
    if hyper_parameters in cums[index].keys():
        new_index = cums[index][hyper_parameters]
        new_trial = tuple(list(hyper_parameters) + [new_index])
        cums[index][new_trial] = trial_summary
        cums[index][hyper_parameters] = new_index + 1
    else:
        cums[index][hyper_parameters] = 1
        new_trial = tuple(list(hyper_parameters) + [0])
        cums[index][new_trial] = trial_summary


def find_alpha2(prob, rets, alphas):
    # in this instance, prob has three dimensions, (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    # rets should have  the two dimensions (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    U_list = []
    for alpha in alphas:
        U = prob * np.log(1.0 + alpha * rets)
        U_list.append(np.sum(U, axis=1))
    U_vec = np.array(U_list)
    # print("----------", U_vec.shape)
    return alphas[np.argmax(U_vec, axis=0)]


"""def find_alpha_for_torch2(prob, rets, device=torch.device('cuda')):
    # in this instance, prob has three dimensions, (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    # rets should have  the two dimensions (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
    alphas = np.array([-1.25, -1.,-0.75,-0.5,-0.25, -0.10 ,0.0,0.10, 0.25,0.5,0.75,1.0,1.25])
    U_list = []
    for alpha in alphas:
      U = prob*torch.log(1.0+alpha*rets)
      U_list.append(U.sum(dim=1).unsqueeze(0))
    U_vec = torch.cat(U_list, dim=0)
    #print("----------", U_vec.shape)
    return alphas[torch.argmax(U_vec, dim=0)]


def find_alpha3(likelihoods, rets):
  # likelihoods has the 2 dimensions, (batch_size, num_of_distribution_bins)
  # rets has the same dimensions.
  from scipy.optimize import minimize
  def loss(alpha):
    return np.sum(likelihoods*np.log(1.0+np.expand_dims(alpha, axis=1)*rets))
  #for likelihood, ret in zip(likelihoods, rets):
    #def loss(alpha):
    #  return - np.dot(likelihood, np.log(1 + ret*alpha[0]))
    #alphas.append(minimize(loss, [0]).x[0])
  ans = minimize(loss, np.ones([likelihoods.shape[0]]), method='Powell').x
  print(ans)
  print(ans.shape)
  input('here it is your guy')
  return np.array(alphas)

# Do not use this, I just realize it is taking
def find_alpha4(probs, rets):
  # in this instance, prob has three dimensions, (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
  # rets should have  the two dimensions (num_of_securities, num_of_a_single_security_data_points, num_of_distribution_bins)
  alphas = np.array([-1.,-0.75,-0.5,-0.25,-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2,0.25,0.5,0.75,1.0])
  official_alphas = []
  for prob, ret in zip(probs, rets):
    U_list = []
    for alpha in alphas:
      U = prob*np.log(1.0+alpha*ret)
      U_list.append(np.sum(U))
    U_vec = np.array(U_list)
    official_alphas.append(alphas[np.argmax(U_vec)])
  return np.array(official_alphas)

"""


def comparison(pos1, pos2, likelihoods, rets):
    # likelihoods has the 2 dimensions, (batch_size, num_of_distribution_bins)
    # rets has the same dimensions.
    z = 0
    bad_times = 0
    for likelihood, ret, po1, po2 in zip(likelihoods, rets, pos1, pos2):
        z = z + 1
        loss1 = - np.dot(likelihood, np.log(1 + ret * po1))
        loss2 = - np.dot(likelihood, np.log(1 + ret * po2))
        if (loss1 > loss2):
            bad_times = bad_times + 1
    print('be aware of the bad times', bad_times)
    input("see bad times")
    # print("this is the optimal alpha", alpha)
    return


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def computeGCD(x, y):

    if x > y:
        small = y
    else:
        small = x
    for i in range(1, small + 1):
        if((x % i == 0) and (y % i == 0)):
            gcd = i

    return gcd
