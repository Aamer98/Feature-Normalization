import torch
import numpy as np

def adjust_learning_rate(optimizer, epoch, lr=0.01, step1=30, step2=60, step3=90):
    """
    Adjusts the learning rate of the optimizer according to a predefined schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        epoch (int): The current epoch number.
        lr (float, optional): The initial learning rate. Default is 0.01.
        step1 (int, optional): The first epoch at which to decay the learning rate by a factor of 10. Default is 30.
        step2 (int, optional): The second epoch at which to decay the learning rate by a factor of 100. Default is 60.
        step3 (int, optional): The third epoch at which to decay the learning rate by a factor of 1000. Default is 90.

    The learning rate is decayed as follows:
        - If epoch >= step3: lr = lr * 0.001
        - Elif epoch >= step2: lr = lr * 0.01
        - Elif epoch >= step1: lr = lr * 0.1
        - Else: lr remains the same
    """
    if epoch >= step3:
        lr = lr * 0.001
    elif epoch >= step2:
        lr = lr * 0.01
    elif epoch >= step1:
        lr = lr * 0.1
    # Else, lr remains unchanged

    # Update the learning rate for all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def one_hot(y, num_class):
    """
    Converts a tensor of class indices into a one-hot encoded tensor.

    Args:
        y (torch.Tensor): Tensor of class indices with shape (n_samples,).
        num_class (int): The total number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor with shape (n_samples, num_class).
    """
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def sparsity(cl_data_file):
    """
    Computes the average sparsity of data across all classes.

    Args:
        cl_data_file (dict): A dictionary where each key is a class label and each value is a list of data arrays for that class.

    Returns:
        float: The mean sparsity (mean number of non-zero elements) across all classes.
    """
    class_list = cl_data_file.keys()
    cl_sparsity = []

    for cl in class_list:
        # Calculate sparsity for each data array in the class
        class_sparsity = [np.sum(x != 0) for x in cl_data_file[cl]]
        # Compute the mean sparsity for the class
        cl_sparsity.append(np.mean(class_sparsity))

    # Return the average sparsity across all classes
    return np.mean(cl_sparsity)
