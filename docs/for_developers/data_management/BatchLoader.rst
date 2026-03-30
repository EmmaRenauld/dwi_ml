.. _batch_loaders:

Batch loader
============

These classes define how batches of streamlines are loaded from a MultiSubjectDataset and how data augmentation is applied. Two main types of batch loaders are implemented:

DWIMLStreamlinesBatchLoader
---------------------------

Loads augmented streamlines only (no MRI volumes). Loads streamlines from a dataset, applies optional data augmentation (resampling, cutting, reversing, noise), and returns them in voxel/corner space.

Methods:
    - ``set_context``: Sets whether the loader operates on training or validation data. Also configures which noise augmentation applies.
    - ``load_batch_streamlines(streamline_ids_per_subj)``: Loads the streamlines for each subject, applies: resampling or compression, splitting, reversing, conversion to voxel + corner coordinates

DWIMLBatchLoaderOneInput
------------------------

Child class of DWIMLStreamlinesBatchLoader. Additionnally communicates with the model to prepare input volume(s) under each point of each streamline and performs trilinear interpolation, and, optionnally, neighborhood extraction.

Methods:
    - ``load_batch_inputs(batch_streamlines, ids_per_subj)``

Note: Must be used with a model with inputs, uses the model’s method: ``prepare_batch_one_input()``.