import logging

__all__ = ['create_logger']

def create_logger(fname, logger_name):
    """
    Creates a logger with a specified name and logs messages to a file.

    Args:
        fname (str): The filename where the logs will be written.
        logger_name (str): The name of the logger.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Get a logger with the specified name
    logger = logging.getLogger(logger_name)

    # Create a file handler for logging
    handler = logging.FileHandler(fname)

    # Define the format for logging messages
    formatter = logging.Formatter('%(levelname)s %(message)s')

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    return logger
