# docs complete
try:
    from train import get_trained_model
    from validation_rnn import rnn_validate_and_collect_performance_metrics
    from validation_seq_to_seq import seq_to_one_validate_and_collect_performance_metrics
    from helpers import hyper_parameters_iterator
    from prepare import get_prepared_data
except:
    print('notebook_style')
import pprint


def one_trial(validation_data_sets, data_loaders, data_preparation_hyper_parameters,
              model_hyper_parameters, real_returns=None, cums=None):
    # validation_data_sets is a list with 2 elements, the first is of the training and second is testing.
    # where both have the following dimension: (num_securities, num_data_points, num_features)

    # data_loaders is a list with 2 elements, both for the learning phase, the first is of the training, and second is of testing.
    # where both have load objects with the following dimensions: (batch_size, seq_len, num_features)

    # distributions is a list of the 2 elements both with dimensions num_bins
    # the first element is the bins distribution of the of the train_train_set.
    # the second is the bins distribution of the validation_test_set.

    # this method trains the model and validates it.

    # returns the trial_summary of the trial with the performance metrics and statistics of the model.

    model = get_trained_model(data_loaders,
                              data_preparation_hyper_parameters,
                              model_hyper_parameters)

    model = model.eval()

    if model_hyper_parameters['type_net'] == 'LSTM':
        return rnn_validate_and_collect_performance_metrics(
            model, validation_data_sets,
            real_returns,
            data_preparation_hyper_parameters,
            model_hyper_parameters, cums)
    else:
        return seq_to_one_validate_and_collect_performance_metrics(
            model, validation_data_sets,
            real_returns,
            data_preparation_hyper_parameters,
            model_hyper_parameters, cums)


def multiple_trials(num_trials, data_preparation_hyper_parameter_list,
                    model_hyper_parameters_list, cums=None):
    # num_trials is the amount of times you want to train and validate the models with the same hyper_parameters.
    # go to hyper_parameters.txt
    # cums is a list of the information you want to accumulate about of each trial.
    # In this application the cums are the performance_statistics, positions and returns.
    for data_preparation_hyper_parameter in hyper_parameters_iterator(data_preparation_hyper_parameter_list):
        print("STARTING data preparation " + 'with \n')
        pprint.pprint(data_preparation_hyper_parameter)
        validation_data_sets, data_loaders, real_returns = get_prepared_data(data_preparation_hyper_parameter)
        for model_hyper_parameters in hyper_parameters_iterator(
                model_hyper_parameters_list):
            print("STARTING the training" + ' with \n')
            pprint.pprint(model_hyper_parameters)
            for num_trial in range(num_trials):
                print("STARTING trial " + str(num_trial))
                if data_preparation_hyper_parameter['skipping_errors'] is True:
                    try:
                        one_trial(validation_data_sets, data_loaders, data_preparation_hyper_parameter, model_hyper_parameters,
                                  real_returns=real_returns, cums=cums)
                    except:
                        print('there was an error with the following parameters')
                        print(data_preparation_hyper_parameter)
                        print(model_hyper_parameters)
                else:
                    one_trial(validation_data_sets, data_loaders, data_preparation_hyper_parameter, model_hyper_parameters,
                              real_returns=real_returns, cums=cums)
