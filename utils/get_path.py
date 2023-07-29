import os
import logging
import datetime
from pathlib import Path


def get_checkpoint_path(output_dir):
    """Get save path
    Args
        output_dir: output_dir file name
        save_path : model save path
    """
    save_path = Path(output_dir)
    save_path = save_path / 'checkpoints'
    logging.info('=> will save model to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_logger_path(output_dir):
    """Get save path
    Args
        output_dir: output_dir file name
        save_path : model save path
    """
    save_path = Path(output_dir)
    save_path = save_path / 'logger'
    logging.info('=> will save logger to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_predictions_path(output_dir):
    """Get save path
    Args
        output_dir: output_dir file name
        save_path : model save path
    """
    save_path = Path(output_dir)
    save_path = save_path / 'predictions'
    logging.info('=> will save predictions to {}'.format(save_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_path_with_time(exper_name='', date=True):
    """Get summary writer path
    Args:
        task(str) : name of task
        exper_name : model save dir name in log
        date : weather print data in output file name
    Return:
        summary writer path
    """
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return exper_name + str_date_time