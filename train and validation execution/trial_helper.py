# doc complete.
import numpy as np
try:
    from helpers import add_info
    from prepare import loadTickers
except:
    print('notebook_style')


def trial_summary_extractor(pos_list, pnl_list, aggregates):
    # validaiton_num_of_securities - self explanatory.
    # pos_list is the list of positions that were taken, it should have dimensions like (validation_num_securities, num_data_points).
    # pnl_list the returns each security and each day positions, dimensions: (validation_num_securities, num_data_points).
    # the aggregates is a list of aggregates that were aggregated during the trial.
    validation_num_of_securities = pos_list[0].shape[0]
    """ part1 extracting the aggregates """
    aggregate_losses = aggregates[0]
    aggregate_MSE_divergences = aggregates[1]
    aggregate_KL_divergences = aggregates[2]

    losses = np.array([sum(aggregate_losses[0]) / len(aggregate_losses[0]), sum(aggregate_losses[1]) / len(aggregate_losses[1])])

    """ part 2 MSE, KL statistics"""
    MSE_Train_mean = aggregate_MSE_divergences[0].mean()
    MSE_Test_mean = aggregate_MSE_divergences[1].mean()
    MSE_mean = np.array([MSE_Train_mean, MSE_Test_mean])

    KL_Train_mean = aggregate_KL_divergences[0].mean()
    KL_Test_mean = aggregate_KL_divergences[1].mean()
    KL_mean = np.array([KL_Train_mean, KL_Test_mean])

    """ part 3 extracting average and variance of the positions"""
    expected_value = np.array([[0., 0.], [0., 0.]])
    variance = np.array([[0., 0.], [0., 0.]])
    all_prob = np.array([[0., 0.], [0., 0.]])

    for train_or_test in range(2):
        for bottom_or_top in range(2):
            if bottom_or_top == 0:
                filtered_positions = (pos_list[train_or_test] < 0)
            else:
                filtered_positions = (pos_list[train_or_test] >= 0)
            all_prob[train_or_test][bottom_or_top] = np.nanmean(filtered_positions)
            expected_value[train_or_test][bottom_or_top] = (pos_list[train_or_test])[filtered_positions].mean()
            variance[train_or_test][bottom_or_top] = (pos_list[train_or_test])[filtered_positions].std()

    all_probs_bottom = all_prob[:, 0]
    all_probs_top = all_prob[:, 1]
    expected_value_bottom = expected_value[:, 0]
    expected_value_top = expected_value[:, 1]
    variance_bottom = variance[:, 0]
    variance_top = variance[:, 1]

    """ part 4 extracting the performance metrics """

    Sharp_per_security = [0., 0.]
    Sharp_per_security[0] = (np.mean(pnl_list[0], axis=1) / np.std(pnl_list[0], axis=1))
    Sharp_per_security[1] = (np.mean(pnl_list[1], axis=1) / np.std(pnl_list[1], axis=1))

    train_sharp_portifolio = (pnl_list[0].sum(axis=0) / validation_num_of_securities)
    test_sharp_portifolio = (pnl_list[1].sum(axis=0) / validation_num_of_securities)

    sharpe_of_average_its = train_sharp_portifolio.mean() / train_sharp_portifolio.std()
    sharpe__of_average_ofs = test_sharp_portifolio.mean() / test_sharp_portifolio.std()
    sharpe_of_average = np.array([sharpe_of_average_its, sharpe__of_average_ofs])

    train_second_half_division = int(len(train_sharp_portifolio) / 2)
    test_second_half_division = int(len(test_sharp_portifolio) / 2)
    sharpe_of_average_second_half_its = train_sharp_portifolio[train_second_half_division:].mean() / train_sharp_portifolio[train_second_half_division:].std()
    sharpe_of_average_second_half_ofs = test_sharp_portifolio[test_second_half_division:].mean() / test_sharp_portifolio[test_second_half_division:].std()
    sharpe_of_average_second_half = np.array([sharpe_of_average_second_half_its, sharpe_of_average_second_half_ofs])

    rets_of_average_its = np.sum(train_sharp_portifolio)
    rets_of_average_ofs = np.sum(test_sharp_portifolio)
    rets_of_average = np.array([rets_of_average_its, rets_of_average_ofs])

    average_sharpe_its = np.nanmean(Sharp_per_security[0])
    average_sharpe_ofs = np.nanmean(Sharp_per_security[1])
    average_sharp = np.array([average_sharpe_its, average_sharpe_ofs])

    average_rets_its = np.mean(pnl_list[0])
    average_rets_ofs = np.mean(pnl_list[1])
    average_rets = np.array([average_rets_its, average_rets_ofs])

    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1]) # main axes
    # ax.set_title('num_iterations = ' + str(iterator_) +  ', seq_length = ' + str(seq_length) + ', batch_size = ' + str(batch_size) + ', lr = ' + str(lr))
    # plt.bar(np.arange(d),prob)

    """ part 5 wrap up and printing the trial summary """
    trial_summary = {"sharpe_of_average": sharpe_of_average, "rets_of_average": rets_of_average,
                     "average_sharp": average_sharp, "average_rets": average_rets,
                     "sharpe_of_average_second_half": sharpe_of_average_second_half,
                     "losses": losses, "MSE_mean": MSE_mean,
                     "KL_mean": KL_mean, "alpha_probs_bottom": all_probs_bottom, "alpha_probs_top": all_probs_top,
                     "expected_value_bottom": expected_value_bottom, "expected_value_top": expected_value_top,
                     "variance_top": variance_top, "variance_bottom": variance_bottom}
    show_partial_results = True
    if show_partial_results:
        print("Showing partial results")
        print(trial_summary)
        print("Done with the partial results")
    return trial_summary


def save_cumulative_data(data_preparation_hyper_parameters, model_hyper_parameters, cums, trial_summary, pnl_list, pos_list):
    # all inputs have been described previously in the documentation.
    # just saves the trial summary, pnl_list, pos_list into cums.

    import pandas as pd
    validation_tickers_names_file = data_preparation_hyper_parameters['validation_tickers_names_file_path']
    validation_tickers = loadTickers(validation_tickers_names_file)

    """  Now we add the zeros in the end as well becuase a few returns were removed due to the fact that they """
    """  the amount of data points has to be divisible by seq_length """
    remainder = data_preparation_hyper_parameters['remainder_of_test_set']
    validation_num_of_securities = pnl_list[1].shape[0]
    pos_list[1] = np.concatenate((pos_list[1], np.zeros([validation_num_of_securities, remainder])), axis=1)
    pnl_list[1] = np.concatenate((pnl_list[1], np.zeros([validation_num_of_securities, remainder])), axis=1)

    """ In this part we set the labels of the pnl dataframe and the position dataframe """
    """ to be the name of each ticker. """
    out_sample_pnl = pd.DataFrame(pnl_list[1])
    out_sample_pnl.index = validation_tickers
    out_sample_pos = pd.DataFrame(pos_list[1])
    out_sample_pos.index = validation_tickers

    """ Extracting the sharp ratios insample """
    this_must_be_the_place = int(pnl_list[0].shape[0] / 2)
    second_half_sharp_in_sample_per_ticker = pnl_list[0][:, this_must_be_the_place:].mean(axis=1) / pnl_list[0][:, this_must_be_the_place:].std(axis=1)
    second_half_sharp_in_sample_per_ticker = pd.DataFrame(second_half_sharp_in_sample_per_ticker)
    second_half_sharp_in_sample_per_ticker.index = validation_tickers

    """ Extract the hyper_parameters that will be index of each cum"""
    analyzed_names = cums[0]['analyzed_names']
    print('analyzed_names', analyzed_names)
    analyzed_hyper_parameters = []
    for name in analyzed_names:
        if name in model_hyper_parameters.keys():
            analyzed_hyper_parameters.append(model_hyper_parameters[name])
        else:
            analyzed_hyper_parameters.append(data_preparation_hyper_parameters[name])

    add_info(cums, 0, analyzed_hyper_parameters, trial_summary)
    add_info(cums, 1, analyzed_hyper_parameters, out_sample_pnl)
    add_info(cums, 2, analyzed_hyper_parameters, out_sample_pos)
    add_info(cums, 3, analyzed_hyper_parameters, second_half_sharp_in_sample_per_ticker)
