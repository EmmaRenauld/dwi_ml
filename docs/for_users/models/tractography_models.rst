.. _tractography_models:

Tractography models
===================

For more explanation on how to use models for tracking, see :ref:`user_tracking`.

TractographyTransformers (tt)
*****************************

This uses transformers and should be the subject of an upcoming publication.


        .. image:: /_static/images/Transformers.png
            :align: center
            :width: 600


To use this model, run script `tt_track_from_model.py`. . To learn more, run::

    tt_track_from_model --help


Learn2track (l2t)
*****************

This is a refactored version of the code prepared by authors of `Poulin2017 <https://link.springer.com/chapter/10.1007/978-3-319-66182-7_62>`_.

        .. image:: /_static/images/Learn2track.png
            :align: center
            :width: 500

To use this model, run script `l2t_track_from_model`. To learn more, run::

    l2t_track_from_model --help

