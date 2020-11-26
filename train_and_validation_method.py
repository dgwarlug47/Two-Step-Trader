import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

quiet = False


def train(model, train_loader, optimizer, epoch, grad_clip=None):
    model.train()
    steps_during_epoch = []
    train_losses = []
    showing_path = False
    prev_parameters = parameter_flattening(model)
    initial_parameters = prev_parameters
    for x in train_loader:
        if model.device == torch.device('cuda'):
            x = x.cuda().contiguous()
        else:
            x = x.contiguous()
        loss = model.loss(x, epoch=epoch)
        optimizer.zero_grad()
        loss.backward()
        # gradients = torch.empty([0]).to(model.device)
        # print('Starting again')
        # for param in model.lstm.parameters():
        #  print(abs(param.grad).mean())
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        now_parameters = torch.zeros([0]).cuda()
        if showing_path:
            now_parameters = parameter_flattening(model)
            steps_during_epoch.append(abs(now_parameters - prev_parameters).sum().item())
        prev_parameters = now_parameters
        train_losses.append(loss.item())
    if showing_path:
        plt.figure()
        plt.plot(steps_during_epoch)
        plt.title('epoch = ' + str(epoch) + ', mean = ' + str(np.array(steps_during_epoch).mean()))
        plt.ylabel('step size')
        plt.xlabel('steps in epochs')
        plt.show()
    final_parameters = parameter_flattening(model)
    return train_losses, abs(final_parameters - initial_parameters).mean().item()


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            if model.device == torch.device('cuda'):
                x = x.cuda().contiguous()
            else:
                x = x.contiguous()
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, model_hyper_parameters):
    epochs, lr = model_hyper_parameters['epochs'], model_hyper_parameters['lr']
    grad_clip = model_hyper_parameters['grad_clip']
    using_swag = model_hyper_parameters['using_swag']
    epochs_or_steps = model_hyper_parameters['epochs_or_steps']
    new_lr = model_hyper_parameters['new_lr']
    start = model_hyper_parameters['start']
    freq = model_hyper_parameters['freq']
    momentum = model_hyper_parameters['momentum']
    adaptation = model_hyper_parameters['adaptation']

    if model_hyper_parameters['optimizer_type'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), betas=(momentum, adaptation), lr=lr)
    if model_hyper_parameters['optimizer_type'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)

    if using_swag:
        import torchcontrib
        if epochs_or_steps:
            opt = torchcontrib.optim.SWA(optimizer, swa_lr=new_lr)
        else:
            opt = torchcontrib.optim.SWA(optimizer, swa_start=start, swa_freq=freq, swa_lr=new_lr)
    else:
        opt = optimizer

    train_losses = []
    mean_steps = []
    test_losses = [eval_loss(model, test_loader)]
    for epoch in range(epochs):
        model.train()
        new_train_loss, new_mean_step = train(model, train_loader, opt, epoch, grad_clip)
        train_losses.extend(new_train_loss)
        mean_steps.append(new_mean_step)
        if using_swag and epochs_or_steps:
            if epoch >= start:
                if epoch - start % freq == 0:
                    opt.update_swa()
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
    if not (np.mean([]) in mean_steps):
        plt.figure()
        plt.plot(np.arange(epochs), mean_steps, label='steps graph')
        plt.ylabel('step size')
        plt.xlabel('epoch')
        plt.show()
    if using_swag:
        opt.swap_swa_sgd()
    return train_losses, test_losses


def parameter_flattening(model):
    new_params = torch.zeros([0]).to(model.device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            new_params = torch.cat((new_params, param.data.clone().reshape(-1)))
    return new_params
