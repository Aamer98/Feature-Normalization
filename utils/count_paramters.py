# Ported from https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/utils.py

__all__ = ['parameter_count']

def parameter_count(module, verbose=False):
    """
    Computes the total number of parameters in a PyTorch module.

    Args:
        module (torch.nn.Module): The PyTorch module to analyze.
        verbose (bool, optional): If True, prints a detailed breakdown of parameters. Default is False.

    Returns:
        int: The total number of parameters in the module.
    """
    params = list(module.named_parameters())
    total_count = sum(int(param.numel()) for name, param in params)
    
    if verbose:
        lines = [
            "",
            "List of model parameters:",
            "=========================",
        ]

        row_format = "{name:<40} {shape:>20} = {total_size:>12,d}"
        
        for name, param in params:
            lines.append(row_format.format(
                name=name,
                shape=" × ".join(str(p) for p in param.size()),
                total_size=param.numel()
            ))
        lines.append("=" * 75)
        lines.append(row_format.format(
            name="All parameters",
            shape="Sum of above",
            total_size=total_count
        ))
        lines.append("")
        print("\n".join(lines))
    
    return total_count
