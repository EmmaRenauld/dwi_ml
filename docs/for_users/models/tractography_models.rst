.. _tractography_models:

Using tractography models
=========================

In both cases, the input data must be formatted as an hdf5. See :ref:`hdf5_usage` for more information.


Hdf5 preparation for our two tractography models
************************************************

Examples below suppose that your hdf5 has the following properties:

    - A subject $subj on which you want to run the tractography.
    - An hdf5 volume group called 'inputs', which contains the input data for the model
    - An hdf5 volume group called 'seeding_mask', which contains a binary mask for the seeds placement (ex, the WM-GM interface).
    - An hdf5 volume group called 'tracking_mask', which contains a binary mask for the tractography (ex, the white matter).

The config file below can be used for Learn2track and TractoTransformer::

    {
        "input": {
            "type": "volume",
            "files": ["anat/*__T1w.nii.gz", "dwi/*__fa.nii.gz", "dwi/*__fodf.nii.gz"],
            "standardization": "per_file",
            "std_mask": ["masks/*__brain_mask.nii.gz"]
        },
        "wm_mask": {
            "type": "volume",
            "files": ["masks/*__mask_wm.nii.gz"],
            "standardization": "none"
        },
        "interface_mask": {
            "type": "volume",
            "files": ["other_masks/*__interface.nii.gz"],
            "standardization": "none"
        },
    }

Some options are common to both models:

    - ``--algo``: Some models not support probabilistic tractography (ex, regression models). Value 'det' should be used. In other cases, the value 'prob' is available.
    - ``--use_gpu``: If your computer has a GPU, we strongly recommand using it. Then, you may also track many streamlines at once for an efficient usage of the model. Use option simultaneous_tracking. For instance, with 10Gb GPU, we could launch ~500 streamlines at the time.
    - ``--tracking_mask``: This option is facultative if you use ``--eos_stop``.
    - ``--help``: Use the help to learn more about other options.

TractoTransformers (tt)
***********************

This uses transformers and should be the subject of an upcoming publication. Its name, TractoTransformer, reflects that this model is similar to the one proposed in `Waizman et al., 2025 <https://arxiv.org/abs/2509.16429>`_. Others have also published Transformer models for Tractography, but did not name their model.


        .. image:: /_static/images/Transformers.png
            :align: center
            :width: 600


To use this model, run script ``tt_track_from_model.py``. For instance::

      tt_track_from_model -f -v \
          --algo det --min_length 10 --max_length 200 \
          --npv 1 --use_gpu --simultaneous_tracking 500 --discard_last \
          --tracking_mask_group wm_mask --eos_stop 0.5  --hdf5 $$hdf5_file \
          $experiment_folder $tracking_subj input out_tractogram.trk \
          interface_mask


Learn2track (l2t)
*****************

This is a refactored version of the code prepared by authors of `Poulin2017 <https://link.springer.com/chapter/10.1007/978-3-319-66182-7_62>`_.

        .. image:: /_static/images/Learn2track.png
            :align: center
            :width: 500

To use this model, run script ``l2t_track_from_model``. For instance::

      l2t_track_from_model -f -v \
          --algo det --min_length 10 --max_length 200 \
          --npv 1 --use_gpu --simultaneous_tracking 500 --discard_last \
          --tracking_mask_group wm_mask --eos_stop 0.5  --hdf5 $$hdf5_file \
          $experiment_folder $tracking_subj input out_tractogram.trk \
          interface_mask
