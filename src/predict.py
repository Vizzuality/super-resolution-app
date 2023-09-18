import os
import json

import numpy as np

from utils import normalize_01, normalize_m11, denormalize_01, denormalize_m11 


class Predictor:
    """
    Predictions with Deep Learning models.
    ----------
    folder_path: string
        Path to the folder with the parameters created during TFRecords' creation.
    dataset_name: string
        Name of the folder with the parameters created during TFRecords' creation.
    model_name: string
        Name of the model
    models: Model
        List of Keras models
    """
    def __init__(self, folder_path, dataset_name, model):
        self.folder_path = folder_path
        self.dataset_name = dataset_name
        self.model = model
        with open(os.path.join(folder_path, dataset_name, "dataset_params.json"), 'r') as f:
            self.params = json.load(f)


    def predict(self, input_array, norm_range=[[0,1], [-1,1]]):
        """
        Predict output.
        Parameters
        ----------
        norm_range: list
            List with two values showing the normalization range.
        """
        # Normalize input image
        if norm_range[0] == [0,1]:
            input_array = normalize_01(input_array)
        elif norm_range[0] == [-1,1]:
            input_array = normalize_m11(input_array)
        else:
            raise ValueError(f'Normalization range should be [0,1] or [-1,1]')

        # Predict
        prediction = self.model.predict(input_array[:,:,:,:3])

        # Display predicted image on map
        # Denormalize output image
        if norm_range[1] == [0,1]:
            prediction = denormalize_01(prediction)
        elif norm_range[1] == [-1,1]:
            prediction = denormalize_m11(prediction)

        return prediction
