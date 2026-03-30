.. _trainers:

Training your model
===================

If your model fits well with our structures and does not have specific needs, our Trainers should already be sufficient for you.

Advantages of using our trainers
--------------------------------

- **Checkpoints**: Our trainers save the model state at each epoch if it is the best one so far, in a folder best_model. They also always save the model state and optimizer state in a checkpoint. This way, if anything happens and your training is stopped, you can continue training from the latest checkpoint.

- **Data Management**: Our trainers know how to interact with your data in the HDF5 and your model. For instance, it can use the BatchSampler to sample streamlines at each batch, and the BatchLoader to interpolate the diffusion data at each coordinate. This way, your model class stays as simple as possible, purely AI-based layers, without the rigmarole and shenanigans of data management.

- **Logs and visu**: They save many metrics as logs on your computer, which you can visualize with our scripts. It also sends data to comet.ml. See :ref:`visu_logs` for more information.

- **Heavy data - ready**: They can manage GPU usage and selecting sampling to limit the loading of heavy data.

- **Training options**: They prepare torch's optimizer (ex, Adam, SGD, RAdam), define the learning rate, etc.

Overview of the process
-----------------------

This is an example of basic script that you could create to train your model with our trainer. It will require:

- Your model
- An instance of our object ``MultiSubjectDataset``: the Trainer knows how to get data in the hdf5, possibly in a lazy way, and store it in a MultiSubjectDataset. See :ref:`ref_data_containers` for more information.
- An instance of a ``BatchSampler``: the Trainer knows how to sample a list of chosen streamlines for a batch. See :ref:`batch_sampler` for more information.
- An instance of a ``BatchLoader``: the Trainer knows how to load the data using the ``MultiSubjectDataset``, and how to modify the streamlines based on your model's requirements, for instance adding noise or compressing, changing the step size, and reversing or splitting the streamlines. See :ref:`batch_loaders` for more information.

For instance, if you need a dMRI input, your final python script could look like this:

.. code-block:: python
   :linenos:

    # Loading the data, possibly with lazy option
    dataset = MultiSubjectDataset(hdf5_file)
    dataset.load_data()

    # Preparing your model
    model = myModel(args)

    # Preparing the BatchSampler
    batch_sampler = DWIMLBatchIDSampler(
            dataset=dataset, streamline_group_name=streamline_group_name)

    # Preparing the BatchLoader.
    batch_loader = DWIMLBatchLoaderOneInput(
            dataset=dataset, model=model,
            input_group_name=input_group_name,
            streamline_group_name=streamline_group_name)

    # Preparing your trainer
    trainer = DWIMLTrainerOneInput(
            model=model, experiments_path=experiments_path,
            experiment_name=experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader)

    # Run the training!
    trainer.train_and_validate()

Once all objects are ready, the Trainer's method ``train_and_validate`` can be used to iterate on epochs until a maximum number of iteration is reached, or a maximum number of bad epochs based on some loss.


Our choices of trainers
-----------------------

``DWIMLTrainer``
************************

This is the main class. For every batch, it loads the chosen streamlines and uses the model, as explained in section 2 below.

``DWIMLTrainerOneInput``
************************

This trainer additionally loads one volume group and accessed the coordinates at each point of your streamlines, or possibly in a neighborhood at each coordinate. Of note, this is done as a separate step, and not through torch's DataLoaders (see explanation in :ref:`batch_loaders`), because interpolation of data is faster through GPU, if you have access, but DataLoaders always work on CPU.

This trainer is expected to be used with a child of ``ModelWithOneInput`` (see page :ref:`other_main_models`).

``DWIMLTrainerOneInputWithGVPhase``
***********************************

We will soon publish how we have used a new generation-validation phase to supervise our models.


Trainers: the code explained
----------------------------

The Trainer's main method is ``train_and_validate``. It is summarized below.

.. code-block:: python
   :linenos:

   def self.train_and_validate():
       for epoch in range(nb_epochs):
           # 1) set the learning rate
           ...

           # 2) Train
           self.train_one_epoch()

           # 3) Validate
           self.validate_one_epoch()

           # 4) Save the model if it's the best epoch
           if this_is_the_best_epoch:
               ...

           # 5) Save a checkpoint
           self.save_checkpoint()

Other steps managed in this method include creating the torch DataLoader from the data_loaders. The DataLoader's collate_fn will be the sampler's load_batch() method.

The ``train_one_epoch`` method and ``validate_one_epoch`` are similar, but validation excludes back-propagation.

.. code-block:: python
   :linenos:

   def self.train_one_epoch():
       for batch in batches:
           self.run_one_batch()

           # If training: back-prop includes:
           # - clip gradients
           # - update torch's optimizer: self.optimizer.step()
           # - reset torch's gradients: self.optimizer.zero_grad(set_to_none=True)
           self.back_propagation()

Finally, ``run_one_batch`` depends on your model. For instance, in ``DWIMLTrainerOneInput``, it interpolates the input at each point and calls the model:

.. code-block:: python
   :linenos:

   def self.run_one_batch():
        # 1) Send data to GPU if available
        ...

        # 2) Formats the streamlines if required by the model
        # ex: SOS, EOS
        ...

        # 3) Interpolate the input (done in the BatchLoader)
        batch_inputs = self.batch_loader.load_batch_inputs(
            streamlines, ids_per_subj)

        # 4) Data augmentation if required
        streamlines = self.batch_loader.add_noise_streamlines_forward(
            streamlines, self.device)

        # 5) Call the model
        model_outputs = self.model(batch_inputs, streamlines_f)

        # 6) Compute the loss
        mean_loss, n = self.model.compute_loss(model_outputs, targets,
                                               average_results=True)

        return mean_loss, n

If this is not right for you, you can override the DWIMLTrainer and re-code this last method.
