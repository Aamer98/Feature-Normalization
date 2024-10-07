# Specify classes or functions that will be exported
__all__ = ['AverageMeter', 'AverageMeterSet']

class AverageMeterSet:
    """
    A set of AverageMeter instances to track and manage multiple metrics.

    Methods:
        - update(name, value, n=1): Update the meter with the given name.
        - reset(): Reset all meters in the set.
        - values(postfix=''): Get current values of all meters with an optional postfix.
        - averages(postfix='/avg'): Get average values of all meters with an optional postfix.
        - sums(postfix='/sum'): Get sums of all meters with an optional postfix.
        - counts(postfix='/count'): Get counts of all meters with an optional postfix.
    """

    def __init__(self):
        """Initialize the AverageMeterSet with an empty dictionary of meters."""
        self.meters = {}

    def __getitem__(self, key):
        """Get the AverageMeter associated with the given key."""
        return self.meters[key]

    def update(self, name, value, n=1):
        """
        Update the meter with the given name.

        Args:
            name (str): The name of the meter to update.
            value (float): The value to update the meter with.
            n (int, optional): The number of occurrences of the value. Defaults to 1.
        """
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        """Reset all meters in the set."""
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        """
        Get current values of all meters with an optional postfix.

        Args:
            postfix (str, optional): String to append to each meter name. Defaults to ''.

        Returns:
            dict: Dictionary of current values for each meter.
        """
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        """
        Get average values of all meters with an optional postfix.

        Args:
            postfix (str, optional): String to append to each meter name. Defaults to '/avg'.

        Returns:
            dict: Dictionary of average values for each meter.
        """
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        """
        Get sums of all meters with an optional postfix.

        Args:
            postfix (str, optional): String to append to each meter name. Defaults to '/sum'.

        Returns:
            dict: Dictionary of sums for each meter.
        """
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        """
        Get counts of all meters with an optional postfix.

        Args:
            postfix (str, optional): String to append to each meter name. Defaults to '/count'.

        Returns:
            dict: Dictionary of counts for each meter.
        """
        return {name + postfix: meter.count for name, meter in self.meters.items()}

class AverageMeter:
    """
    Computes and stores the average, sum, current value, and count.

    Methods:
        - update(val, n=1): Update the meter with a new value.
        - reset(): Reset the meter statistics.
    """

    def __init__(self):
        """Initialize the AverageMeter and reset its statistics."""
        self.reset()

    def reset(self):
        """Reset all statistics to zero."""
        self.val = 0      # Current value
        self.avg = 0      # Average of all updates
        self.sum = 0      # Sum of all updates
        self.count = 0    # Total number of updates

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The new value to add.
            n (int, optional): The number of times this value occurred. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format_spec):
        """
        Format the current and average values.

        Args:
            format_spec (str): Format specification.

        Returns:
            str: Formatted string showing current and average values.
        """
        return "{self.val:{format_spec}} ({self.avg:{format_spec}})".format(
            self=self, format_spec=format_spec
        )
