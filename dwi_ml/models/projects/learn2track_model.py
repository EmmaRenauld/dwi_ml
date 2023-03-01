# -*- coding: utf-8 -*-
import logging
from typing import Union, List

import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings as \
    keys_to_tensor_embeddings, NoEmbedding
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections, ModelForTracking, ModelWithNeighborhood,
    MainModelOneInput)
from dwi_ml.models.projects.stacked_rnn import StackedRNN

logger = logging.getLogger('model_logger')  # Same logger as Super.


class Learn2TrackModel(ModelWithPreviousDirections, ModelForTracking,
                       ModelWithNeighborhood, MainModelOneInput):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, a RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedding_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 input_embedding_key: str,
                 input_embedding_size: Union[int, None],
                 input_embedding_size_ratio: Union[float, None],
                 # RNN
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool,
                 use_layer_normalization: bool, dropout: float,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Params
        ------
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        input_embedding_key: str
            Key to an embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings).
            Default: 'no_embedding'.
        input_embedding_size: int
            Output embedding size for the input. If None, will be set to
            input_size.
        input_embedding_size_ratio: float
            Other possibility to define input_embedding_size, which then equals
            [ratio * (nb_features * (nb_neighbors+1))]
        rnn_key: str
            Either 'LSTM' or 'GRU'.
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will depend
            on the skip_connection parameter.
        use_skip_connection: bool
            Whether to use skip connections. See [1] (Figure 1) to visualize
            the architecture.
        use_layer_normalization: bool
            Wheter to apply layer normalization to the forward connections. See
            [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        if prev_dirs_embedding_key == 'no_embedding':
            if prev_dirs_embedding_size is None:
                prev_dirs_embedding_size = 3 * nb_previous_dirs
            elif prev_dirs_embedding_size != 3 * nb_previous_dirs:
                raise ValueError("To use identity embedding, the output size "
                                 "must be the same as the input size!")

        super().__init__(
            experiment_name=experiment_name, log_level=log_level,
            # For modelWithNeighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            # For super MainModelWithPD:
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedding_size=prev_dirs_embedding_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # For super ModelForTracking:
            dg_args=dg_args, dg_key=dg_key)

        self.input_embedding_key = input_embedding_key
        self.nb_features = nb_features
        self.dropout = dropout

        # ----------- Checks
        if self.input_embedding_key not in keys_to_tensor_embeddings.keys():
            raise ValueError("Embedding choice for x data not understood: {}"
                             .format(self.embedding_key_x))

        # ---------- Instantiations
        # 1. Previous dirs embedding: prepared by super.
            if embedding_layer is None:
                raise ValueError(
                    "SOS_as_zero_embedding cannot be chosen when no "
                    "embedding layer is defined.")

        # 2. Input embedding
        self.input_size = nb_features * (self.nb_neighbors + 1)
        if input_embedding_size and input_embedding_size_ratio:
            raise ValueError("You must only give one value, either "
                             "input_embedding_size or "
                             "input_embedding_size_ratio")
        elif input_embedding_size_ratio:
            self.input_embedding_size = int(self.input_size *
                                            input_embedding_size_ratio)
        elif input_embedding_size:
            self.input_embedding_size = input_embedding_size
        else:
            self.input_embedding_size = self.input_size

        input_embedding_cls = keys_to_tensor_embeddings[input_embedding_key]
        self.input_embedding = input_embedding_cls(
            input_size=self.input_size, output_size=self.input_embedding_size)

        # 3. Stacked RNN
        rnn_input_size = self.input_embedding_size
        if self.nb_previous_dirs > 0:
            rnn_input_size += self.prev_dirs_embedding_size
        self.rnn_model = StackedRNN(
            rnn_key, rnn_input_size, rnn_layer_sizes,
            use_skip_connection=use_skip_connection,
            use_layer_normalization=use_layer_normalization,
            dropout=dropout)

        # 4. Direction getter:
        self.instantiate_direction_getter(self.rnn_model.output_size)

        # If multiple inheritance goes well, these params should be set
        # correctly
        assert self.forward_uses_streamlines
        assert self.loss_uses_streamlines
        if nb_previous_dirs == 0:
            # Then the forward method will not use the streamlines.
            self.forward_uses_streamlines = False

    @property
    def params_for_checkpoint(self):
        # Every parameter necessary to build the different layers again.
        # during checkpoint state saving.
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'input_embedding_key': self.input_embedding_key,
            'input_embedding_size': int(self.input_embedding_size),
            'input_embedding_size_ratio': None,
            'rnn_key': self.rnn_model.rnn_torch_key,
            'rnn_layer_sizes': self.rnn_model.layer_sizes,
            'use_skip_connection': self.rnn_model.use_skip_connection,
            'use_layer_normalization': self.rnn_model.use_layer_normalization,
            'dropout': self.dropout,
        })

        return params

    def forward(self, inputs: List[torch.tensor],
                target_streamlines: List[torch.tensor] = None,
                hidden_reccurent_states: tuple = None,
                return_state: bool = False, is_tracking: bool = False):
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs: List[torch.tensor]
            Batch of input sequences, i.e. MRI data. Length of the list is the
            number of streamlines in the batch. Each tensor is of size
            [nb_points, nb_features]. During training, should be the length of
            the streamlines minus one (last point is not used). During
            tracking, nb_points should be one; the current point.
        target_streamlines: List[torch.tensor],
            Batch of streamlines. Only used if previous directions are added to
            the model. Used to compute directions; its last point will not be
            used.
        hidden_reccurent_states : tuple
            The current hidden states of the (stacked) RNN model.
        return_state: bool
            If true, return new hidden recurrent state together with the model
            outputs.
        is_tracking: bool
            If False, streamlines contains one more point than inputs, as their
            point does not have a target (a direction). Else, while tracking,
            streamlines have the same number of points as inputs (we do not use
            the targets, so it's ok, we only need to compute the previous
            dirs).

        Returns
        -------
        model_outputs : Tensor
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
            NOTE: this tensor's format will be one direction per point in the
            input, with the same organization os the initial packed sequence.
            It should be compared with packed_sequences's .data.
            Or it is possible to form pack a packed sequence with
            output = PackedSequence(output,
                                   inputs.batch_sizes,
                                   inputs.sorted_indices,
                                   inputs.unsorted_indices)
        out_hidden_recurrent_states : list[states]
            One value per layer.
            LSTM: States are tuples; (h_t, C_t)
                Size of tensors are each [1, nb_streamlines, nb_neurons].
            GRU: States are tensors; h_t.
                Size of tensors are [1, nb_streamlines, nb_neurons].
        """
        try:
            # Apply model. This calls our model's forward function
            # (the hidden states are not used here, neither as input nor
            # outputs. We need them only during tracking).
            model_outputs, new_states = self._run_forward(
                inputs, target_streamlines, hidden_reccurent_states,
                is_tracking)
        except RuntimeError:
            # Training RNNs with variable-length sequences on the GPU can
            # cause memory fragmentation in the pytorch-managed cache,
            # possibly leading to "random" OOM RuntimeError during
            # training. Emptying the GPU cache seems to fix the problem for
            # now. We don't do it every update because it can be time
            # consuming.
            # If it fails again, try closing terminal and opening new one to
            # empty cache better.
            logging.warning("There was a RunTimeError. Emptying cache and "
                            "trying again!")

            torch.cuda.empty_cache()
            model_outputs, new_states = self._run_forward(
                inputs, target_streamlines, hidden_reccurent_states,
                is_tracking)

        if return_state:
            # Tracking
            return model_outputs, new_states
        else:
            # Training
            return model_outputs

    def _run_forward(self, inputs, streamlines: List[torch.Tensor],
                     hidden_reccurent_states, is_tracking):
        # Ordering of PackedSequence for 1) inputs, 2) previous dirs and
        # 3) targets (when computing loss) may not always be the same.
        # Pack inputs now and use that information for others.
        # Shape of inputs.data: nb_pts_total * nb_features
        if is_tracking:
            # We are only dealing with one point per streamlines (and the
            # hidden state). Ensure that packed_sequence has the right size.
            nb_streamlines = len(inputs)
            inputs = torch.vstack(inputs)
            batch_sizes = torch.as_tensor([nb_streamlines])
            sorted_indices = torch.arange(nb_streamlines, device=self.device)
            unsorted_indices = torch.arange(nb_streamlines, device=self.device)
            inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                    unsorted_indices)
        else:
            # Streamlines have different lengths. Packing.
            inputs = pack_sequence(inputs, enforce_sorted=False)
            batch_sizes = inputs.batch_sizes
            sorted_indices = inputs.sorted_indices
            unsorted_indices = inputs.unsorted_indices

        # ==== 1. Previous dirs embedding ====
        if self.nb_previous_dirs > 0:
            dirs = self.prepare_targets_forward(streamlines, during_forward=True)

            # During training, we need all the previous n directions, during
            # tracking, the last one only.
            point_idx = -1 if is_tracking else None

            # Result will be a packed sequence.
            n_prev_dirs_embedded = self.normalize_and_embed_previous_dirs(
                dirs, unpack_results=False, point_idx=point_idx,
                packing_order=(batch_sizes, sorted_indices, unsorted_indices))
        else:
            n_prev_dirs_embedded = None

        # ==== 2. Inputs embedding ====
        # Avoiding unpacking and packing back if not needed.
        if not isinstance(self.input_embedding, NoEmbedding) or \
                self.nb_previous_dirs > 0:

            # Embedding. Shape of inputs: nb_pts_total * embedding_size
            inputs = self.input_embedding(inputs.data)

            # ==== 3. Concat with previous dirs ====
            if n_prev_dirs_embedded is not None:
                inputs = torch.cat((inputs, n_prev_dirs_embedded.data), dim=-1)
                logger.debug("Concatenated shape: {}".format(inputs.shape))

            # Shaping again as packed sequence.
            # Shape of inputs.data: nb_pts_total * embedding_size_total
            inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                    unsorted_indices)

        # ==== 3. Stacked RNN (on packed sequence, returns a tensor) ====
        # rnn_output shape: nb_pts_total * last_hidden_layer_size
        rnn_output, out_hidden_recurrent_states = self.rnn_model(
            inputs, hidden_reccurent_states)

        logger.debug("*** 5. Direction getter....")
        # direction getter can't get a list of sequences.
        # output will be a tensor, but with same format as input.data.
        # we will get a direction for each point.
        model_outputs = self.direction_getter(rnn_output)
        logger.debug("Output size: {}".format(model_outputs.shape[-1]))

        # Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.

        # Not unpacking now; we will compute the loss point by point on this
        # whole tensor. Sending as PackedSequence to be sure that targets will
        # be concatenated in the same order when computing loss.
        if not is_tracking:
            model_outputs = PackedSequence(model_outputs, batch_sizes,
                                           sorted_indices, unsorted_indices)

        return model_outputs, out_hidden_recurrent_states

    def compute_loss(self, model_outputs: PackedSequence,
                     target_streamlines: List[torch.Tensor], **kw):
        """
        Computes the loss function using the provided outputs and targets.
        Returns the mean loss (loss averaged across timesteps and sequences).

        Parameters
        ----------
        model_outputs : PackedSequence
            The model outputs for a batch of sequences. Ex: a gaussian mixture
            direction getter returns a Tuple[Tensor, Tensor, Tensor], but a
            cosine regression direction getter return a simple Tensor. Please
            make sure that the chosen direction_getter's output size fits with
            the target ou the target's data if it's a PackedSequence.
        target_streamlines : List[tensor]
            The target streamlines for the batch.

        Returns
        -------
        mean_loss : torch.Tensor
            The loss between the outputs and the targets, averaged across
            timesteps and sequences.
        """
        # Packing dirs and using the .data instead of looping on streamlines.
        # Anyway, loss is computed point by point.
        target_dirs = pack_sequence(target_streamlines, enforce_sorted=False)
        assert torch.equal(target_dirs.sorted_indices,
                           model_outputs.sorted_indices)

        # Computing loss
        return self.direction_getter.compute_loss(model_outputs.data,
                                                  target_dirs.data)

    def get_tracking_directions(self, model_outputs, algo):
        """
        Params
        ------
        model_outputs: Tensor
            Our model's previous layer's output.
        algo: str
            'det' or 'prob'.

        Returns
        -------
        next_dir: list[array(3,)]
            Numpy arrays with x,y,z value, one per streamline data point.
        """
        return self.direction_getter.get_tracking_directions(
            model_outputs, algo)
