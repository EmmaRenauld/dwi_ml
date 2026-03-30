.. _tracking:

Tracking with your model
========================

For tracking, you may observe how scripts `l2t_track_from_model` or `tt_track_from_model` work. They use two main objects, the Tracker and the Propagator, similarly as in scilpy.

Trackers
--------

- ``DWIMLAbstractTracker``: Performs tractography: starts from a seeding mask, and, at each point, advances one step using the model's output.

- ``DWIMLTrackerFromWholeStreamline``: Child class in cases where we need to send the whole streamline to the model in order to generate the next point's position. We need to copy them in memory here as long as the streamline is not finished being tracked.

- ``DWIMLTrackerOneInput``: Child class where the dMRI input must be interpolated at each point (using the BatchLoader) and sent as input to the model. Can be combined with DWIMLTrackerFromWholeStreamline.


Differences with scilpy
------------------------

If you are familiar with scilpy (`Renauld 2023 <https://apertureneuro.org/article/154022-tractography-analysis-with-the-scilpy-toolbox>`_), you will notice similarity in the code. Here is a comparison of our Tracker to theirs:

- In scilpy, the *theta* parameter defines an aperture cone inside which the next direction can be sampled. Here, sampling is not as straightforward. Ex, in the case of regression, the next direction is directly obtained from the model. Instead, theta is used as a stopping criterion.

- In scilpy, at each propagation step, the propagator uses the local model (ex, DTI, fODF) to decide the next direction. Here, we send data as input to the machine learning model. The model may receive additional inputs as compared to classical tractography (ex, the hidden states in RNNs, or the full beginning of the streamline in Transformers).

- GPU processing: We offer a GPU option, where many streamlines are created simultaneously, to take advantage of the GPU capacities. Our GPU option uses torch, whereas in scilpy, the GPU option uses openCL and a different implementation.
