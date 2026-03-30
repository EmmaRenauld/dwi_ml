.. _batch_sampler:

Batch sampler
=============

These classes defines how to sample the streamlines available in the MultiSubjectData. You are encouraged to contribute to dwi_ml by adding any child class here.


DWIMLBatchIDSampler
-------------------

- Defines the __iter__ method: It finds a list of streamlines ids and associated subjects that you can later load in your favorite way. It limits the number of subjects per batch and orders streamlines by subjects to make sure you don't need to load a full new volume at each new streamline.

