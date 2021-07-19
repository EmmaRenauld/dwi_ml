import logging

import torch


class ModelAbstract(torch.nn.module):
    def __init__(self):
        raise NotImplementedError

    @property
    def hyperparameters(self):
        raise NotImplementedError

    @property
    def attributes(self):
        """All parameters necessary to create again the same model"""
        hyperparameters = self.hyperparameters
        return hyperparameters

    @property
    def state_dict(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


def init_model(**unused_kwargs) -> ModelAbstract:
    """
    Write your own code here. You may check our scil-vital Github
    repository for our lab member's projects, such as Learn2Track.

    Loaded training_dataset is probably necessary to find informations such
    as feature sizes.
    """
    model = ModelAbstract()

    logging.debug('Unused kwargs: {}'.format(unused_kwargs))

    return model
