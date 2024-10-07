# Compute accuracy
import torch

def accuracy(logits, ground_truth, topk=[1,]):
    """
    Computes accuracy metrics including overall accuracy, per-class accuracy, and top-k accuracy.

    Args:
        logits (torch.Tensor): Predicted logits from the model with shape (n_samples, n_classes).
        ground_truth (torch.Tensor): Ground truth labels with shape (n_samples,).
        topk (list[int], optional): List of top-k values for which to compute accuracy. Default is [1].

    Returns:
        dict: A dictionary containing the following keys:
            - 'average' (torch.Tensor): Overall top-k accuracy percentages.
            - 'per_class_average' (torch.Tensor): Average top-k accuracy per class percentages.
            - 'per_class' (list[list[float]]): List containing per-class top-k accuracy percentages.
            - 'gt_unique' (torch.Tensor): Tensor of unique ground truth labels.
            - 'topk' (list[int]): The list of top-k values used.
            - 'num_classes' (int): Number of classes inferred from logits.
    """
    assert len(logits) == len(ground_truth), "Logits and ground truth must have the same number of samples."

    n, d = logits.shape  # n: number of samples, d: number of classes

    # Get unique ground truth labels
    label_unique = torch.unique(ground_truth)
    num_classes = len(label_unique)

    # Initialize accuracy dictionary
    acc = {
        'average': torch.zeros(len(topk)),
        'per_class_average': torch.zeros(len(topk)),
        'per_class': [[] for _ in label_unique],
        'gt_unique': label_unique,
        'topk': topk,
        'num_classes': d,
    }

    max_k = max(topk)
    # Get the indices of the top-k predictions
    argsort = torch.argsort(logits, dim=1, descending=True)[:, :min(max_k, d)]
    # Create a binary matrix indicating whether the top-k predictions match the ground truth
    correct = (argsort == ground_truth.view(-1, 1)).float()

    for idx, label in enumerate(label_unique):
        # Find indices where the ground truth label equals the current label
        indices = torch.nonzero(ground_truth == label, as_tuple=False).view(-1)
        correct_target = correct[indices]

        # Calculate top-k accuracy for each k in topk
        for k_idx, k in enumerate(topk):
            num_correct = torch.sum(correct_target[:, :k]).item()
            acc_partial = num_correct / len(correct_target)
            acc['average'][k_idx] += num_correct
            acc['per_class_average'][k_idx] += acc_partial
            acc['per_class'][idx].append(acc_partial * 100)

    # Calculate average accuracies
    acc['average'] = acc['average'] / n * 100
    acc['per_class_average'] = acc['per_class_average'] / num_classes * 100

    return acc
