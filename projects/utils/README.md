- # data folder
    - #### data_loader.py
        It contains the function get_data_loader(dl_cfg, data_root, split_type, datasets, FAR_FRR=None, augmentation=None, rank=0, num_gpus=1)
        This function load one or more datasets and returns a tuple containing the data loaders and the number of unique IDs in the datasets.
        Args:
        - dl_cfg (dict): Configuration for the data loader, use for example cfg.training.data_loaders.train
        - data_root (str): Root directory of the data.
        - split_type (str): Type of data split (e.g., "train", "val", "test").
        - datasets (dict): Dictionary containing the dataset names and their configurations.
        - FAR_FRR (bool, optional): Whether to use a PairedSampleDataset (FAR/FRR evaluation) orSingleSampleDataset (other evaluations). Defaults to None.
        - augmentation (callable, optional): Data augmentation function. Defaults to None.
        - rank (int, optional): Rank of the current process in distributed training. Defaults to 0. USED ONLY IF num_gpus >1.
        - num_gpus (int, optional): Number of GPUs used for training. Defaults to 1.
        
        Returns:
        - tuple: A tuple containing the data loader and the number of unique IDs in the dataset.
        
    - #### dataset.py
        It contains the classes loaded in the data_loader.py file. The classes PairedSampleDataset and SingleSampleDataset load the dataset in different ways. Look at  __getitem__ to understand how they do it. 
    - #### celeba folder
        - ##### celeba_dataset_analysis.py
            ! NEEDS TO BE REVIEWED TO CHECK IF IT WORKS CORRECTLY
        - ##### celeba_debug_annotation.py
            ! Run the celeba_reshuffle.py script before this.
            ! It needs to be improved to make it more readable.
            This script reads data from an annotation CSV file, extracts a subset of samples for debugging purposes, creates a new debug dataset folder, reshuffles the samples, and saves the reshuffled data and corresponding images to the debug dataset folder.
            The new dataset will be considerably smaller than the original one. To control the size of the dataset change the variable: 
                PERC_IDS_DEBUG: Used to determine the number of IDs in the debug dataset.
        - ##### celeba_reshuffle.py
            the code reads the CelebA dataset annotations, splits the data into various subsets based on specific conditions, assigns split labels to each subset, reshuffles the data, and saves the reshuffled data to a CSV file. The resulting CSV file will contain the annotations for the train, validation, and test sets with different subsets based on ID conditions.
            - split_test_same_ID: is a test set where the IDs in it are the same present in the training and validation dataset.
            - split_test:is a test set where the IDs in it are different from the ones in the training and validation dataset.
            - split_train_a_same_IDs and split_train_b_same_IDs have the images from the same IDs, so the same IDs can be found in both of them. 
            - split_train_a_different_IDs and split_train_b_different_IDs have the images from different IDs, so the IDs in one splits are not present in the other.
            - split_train: Contains the IDs and images from both couples of training splits above.
            - split_val: Contains the IDs BUT NOT images from both couples of training splits above.
    - #### cropping folder
        This folder contains the script used to crop the faces in the original celeba dataset. 
        This code is not currently maintained !
    - #### augmentations folder
        ! NEEDS TO BE REVIEWED TO CHECK IF IT WORKS CORRECTLY
        - ##### custom_augmentations.py
            File where single custom augmentations functions are stored.
        - ##### create_augmentation.py
            Use it to create new .yaml files where to save custom augmentations. These files will be automatically generated in the projects/config/augmentation folder.

- # wandb folder
    - #### upload_experiment_dir.py and download_experiment_dir.py 
        Run these scripts to download or upload from/to WandB the experiments folders.
    - #### download_data.py
        Synchronizes the data by downloading the dataset from WandB repository or the original source.
        - ##### How to use it
            Import the function synchronize_data(root, dataset_type, dataset_version) which has as input the following parameters and returns a pandas Dataframe object:
            Args:
            - root (str): The root directory where the dataset will be downloaded.
            - dataset_type (str): The type of dataset to be downloaded.
            - dataset_version (str): The version of the dataset to be downloaded.
            
            Returns:
            - pandas.DataFrame: The dataframe containing the downloaded dataset.
- # tsne.py - T-SNE Analysis and Visualization Tool
    This code provides a tool for performing T-SNE (t-Distributed Stochastic Neighbor Embedding) analysis on a model's output features and visualizing the results. T-SNE is a dimensionality reduction technique commonly used for visualizing high-dimensional data in a lower-dimensional space.
    - #### How to use it
        Import the function tsne_analysis(model, data_loader, ids_unique, device) which has as input the following parameters and it doesn't return anything back:
        Args:
        - model: The trained model.
        - data_loader: Data loader providing the input data.
        - ids_unique: Unique IDs of the data samples.
        - device: Device to run the model on.
        - plot_size (int, optional): Size of the T-SNE plot. Default is 1000.
        - max_image_size (int, optional): Maximum size of the images in the T-SNE plot. Default is 100.
        
        After the analysis two images representing the results of the analysis are saved in the run folder and uploaded on WandB. 
- # utils.py 
    - #### poly_scheduler.py
        This code defines a class called PolyScheduler, which is a learning rate scheduler for optimizing models. The class inherits from the _LRScheduler class. The code is from https://github.com/deepinsight/insightface
        Args:
        - optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        - base_lr (float): The base learning rate.
        - max_steps (int): The total number of steps in the training process.
        - warmup_steps (int): The number of warm-up steps to gradually increase the learning rate.
        - last_epoch (int, optional): The index of the last epoch. Default is -1.
    - #### set_seed
        sets the seed for the current run. You can set the seed number giving it as input but it is optional.
    - #### EarlyStop - Early Stopping for Training
        This code provides a class EarlyStop that implements early stopping during training based on a validation loss threshold.
        - ##### How to use it
            Init the class EarlyStop with the following parameters:
            Args:
            - tot_epochs (int): Total number of epochs for training.
            - early_stop_cfg (object): Configuration object for early stopping parameters.

            Run the function check_early_stop_condition() to check if the early stopping condition is satisfied.
            Args:
            - val_loss (float): Validation loss value.
            - epoch (int): Current epoch number.

            Return:
            - bool: True if the early stopping condition is met, False otherwise.
    - #### BatchSizeScheduler - Batch Size Scheduler for Training
        The BatchSizeScheduler class is designed to dynamically adjust the batch size during training based on a predefined schedule starting after a minimum number of epochs.
        - ##### How to use it
            Init the class BatchSizeScheduler with the following parameters:
            Args:
            - tot_epochs (int): Total number of epochs.
            - cfg (object): Configuration object containing parameters for batch size scheduling.

            Run the function update_batch_size() to check if the conditions are satisfied.
            Args:
            - epoch (int): Current epoch number.
            - training_cfg (object): Training configuration object.
            - data_loader_func (function): Data loader function.
            - train_aug (object): Training data augmentation object.
            - train_loader (object): Training data loader object.

            Returns:
            - object: Updated training data loader object.
- # visualize_dataset.py
    Visualizes a dataset and creates a html-file of the visualization and opens it in the browser.
    This code is not currently maintained !