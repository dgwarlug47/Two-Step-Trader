# doc complete.
try:
    from helpers import save_training_plot
    from train_and_validation_method import train_epochs
    from AR_Base import AR_Model
except:
    print('notebook_style')


def get_trained_model(data_loaders, data_preparation_hyper_parameters, model_hyper_parameters):
    # data_loaders is a list with 2 elements, both for the learning phase, the first is of the training, and second is of testing.
    # where both have load objects with the following dimensions: (batch_size, seq_len, num_features)

    # train distribution is self explanatory.

    # go to hyper_parameters.txt

    """ part 1 trains the model. """
    device = data_preparation_hyper_parameters['device']

    train_loader = data_loaders[0]
    test_loader = data_loaders[1]

    """ extracst the num_features of the data set, in order to pass it to the model. """
    for x in train_loader:
        model_hyper_parameters['num_features'] = x.shape[2] - 1
        break

    print("Start Learning")
    import time
    time4 = time.clock()
    model = AR_Model(device, data_preparation_hyper_parameters, model_hyper_parameters).to(device)

    """ part 2 plots the training, test curve."""
    train_losses, test_losses = train_epochs(model, train_loader, test_loader, model_hyper_parameters)
    save_training_plot(train_losses, test_losses, 'Train vs Test', './Results')

    time6 = time.clock()
    print("Total Training time, ", time6 - time4)
    print("Start Validation")
    return model
