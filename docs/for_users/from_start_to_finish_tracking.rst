.. _from_start_to_finish_tracking:

Training tracking models: from start to finish
==============================================

If you want, you can use our scripts to train our models with a new set of hyperparameters!

Overview of the process
***********************

No matter the model, the process will probably contain the following steps:

1. **Creating a hdf5 file**.
    Our library works with data in the hdf5 format. See :ref:`hdf5_usage` for more information.

2. **Training the model**.
    At each epoch, the script saves the model state if it is the best one so far, in a folder ``best_model``, but also always saves the model state and optimizer state in a checkpoint. This way, if anything happens and your training is stopped, you can continue training from the latest checkpoint.

    To learn more about training options, see the help. For instance, ``l2t_train_model --help`` or ``tt_train_model --help``.

3. **Visualizing the logs**.
    We have tools to help you supervise the results. See :ref:`visu_logs` for more information.

4. **Visualizing the loss**.
    You can use your favorite .trk visualizer (ex, Mi-Brain) to view the local loss along points of your streamlines.

5. **Using your newly trained model!**
    See :ref:`tractography_models` for more information.


Learn2track (l2t)
*****************

Here is what your bash script will look like for Learn2track. For each step, you will have many options to define!

.. code-block:: bash

    # Create a hdf5 file
    # Most options are given through the config file
    dwiml_create_hdf5_dataset $input_folder $out_file $config_file \
        $training_subjs $validation_subjs $testing_subjs

    # Train a model.
    # Play with options! Here are the mandatory inputs:
    l2t_train_model $saving_path $experiment_name $hdf5_file \
        $input_group_name $streamline_group_name

    # If you want to train your model a little more...
    l2t_resume_training_from_checkpoint $saving_path $experiment_name \
    --new_patience 10 --new_max_epochs 300

    # Visualize the logs
    dwiml_visualize_logs $saving_path/$experiment_name

    # See which points of your training streamlines have the worst loss
    l2t_visualise_loss $saving_path/$experiment_name $hdf5_file $subj $input_group_name \
        --out_prefix colored_tractogram --subset training \
        --use_gpu --batch_size 400 --streamlines_group $streamlines \
        --colormap turbo --show_now --compute_histogram   --save_colored_tractogram \
        --save_colored_best_and_worst 0.5

    # Once happy, use your final model to track from it!
    l2t_track_from_model $saving_path/$experiment_name $subj $input_group $out_tractgram $seeding_mask_group



TractoTransformers (tt)
***********************

Here is watch your bash script will look like for TractoTransformers. For each step, you will have many options to define!

.. code-block:: bash

    # Create a hdf5 file
    # Most options are given through the config file
    dwiml_create_hdf5_dataset $input_folder $out_file $config_file \
        $training_subjs $validation_subjs $testing_subjs

    # Train a model.
    # Play with options! Here are the mandatory inputs:
    tt_train_model $saving_path $experiment_name $hdf5_file \
        $input_group_name $streamline_group_name

    # If you want to train your model a little more...
    tt_resume_training_from_checkpoint $saving_path $experiment_name \
    --new_patience 10 --new_max_epochs 300

    # Visualize the logs
    dwiml_visualize_logs $saving_path/$experiment_name

    # See which points of your streamlines have the worst loss
    tt_visualize_loss $saving_path/$experiment_name $hdf5_file $subj $input_group_name

    # Once happy, use your final model to track from it!
    tt_track_from_model $saving_path/$experiment_name $subj $input_group $out_tractgram $seeding_mask_group

    # Visualize where the attention focuses for your tractogram!
    tt_visualize_weights $saving_path/$experiment_name $hdf5_file $subj $input_group_name $out_tractogram


