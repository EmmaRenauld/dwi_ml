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

    def save(self):
        raise NotImplementedError

    def run_model_and_compute_loss(self, data,
                                   cpu_computations_were_avoided: bool = True,
                                   is_training: bool = False) -> float:
        """Run a batch of data through the model (calling its forward method)
        and return the mean loss. If training, run the backward method too.

        Parameters
        ----------
        data : Any
            This is the output of your sampler's load_batch() function.
        cpu_computations_were_avoided: bool
            Batch sampler's avoid_cpu_computation option value at the moment of
            loading the batch data.
        is_training : bool
            If True, record the computation graph and backprop through the
            model parameters.

        Hint: If your sampler was instantiated with avoid_cpu_computations,
        you need to deal with your data accordingly here!
        Use the sampler's self.load_batch_final_step method.
        """
        raise NotImplementedError
