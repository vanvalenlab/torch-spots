import numpy as np
import torch

from huggingface_hub import hf_hub_download

from torch_spots.detection.dotnet import SpotNet
from skimage.feature import peak_local_max

import numpy as np
import glob

from pathlib import Path

from torch_spots.detection.utils import spotnet_preprocess, tile_input, untile_output, \
      max_cp_array_to_point_list_max, extract_spots_prob_from_coords_maxpool


class SpotDetection():

    def __init__(
            self,
            model_path=None,
            device=None
    ):
        
        self.model = SpotNet()

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        if model_path is None:
            
            hf_hub_download(repo_id='vanvalenlab/torch-spots', 
                            filename='torch-spots_2026-07-29.pth',
                            local_dir=Path.home() / '.deepcell/models')
            model_path = Path.home() / '.deepcell/models/torch-spots_2026-07-29.pth'
            
        self.model_path = model_path
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.eval().to(self.device)
        self.input_shape = 128

    def _y_annotations_to_point_list_max_offsets(
            self, 
            y_pred, 
            threshold=0.95, 
            min_distance=2, 
            x_offsets=0, 
            y_offsets=0
    ):
        """Convert raw prediction to a predicted point list using
        ``skimage.feature.peak_local_max`` to determine local maxima in classification
        prediction image, and their corresponding regression values will be used to
        create a final spot position prediction which will be added to the output spot
        center coordinates list.

        Args:
            y_pred (array): a dictionary of predictions with keys `'classification'` and
                `'offset_regression'` corresponding to the named outputs of the
                ``dot_net_2D model``.
            threshold (float): a number in ``[0, 1]``. Pixels with classification
                score > `threshold` are considered as containing a spot center.
            min_distance (float): the minimum distance between detected spots in pixels.

        Returns:
            array: spot center coordinates of the format [[y0, x0], [y1, x1],...]
        """
        if not isinstance(y_pred, dict):
            raise TypeError('Input predictions must be a dictionary.')

        dot_centers = []

        for ind in range(np.shape(y_pred['detections'])[0]):
            # iterate through individual images in this batch

            dot_pixel_inds = peak_local_max(y_pred['detections'][ind, 1],
                                            min_distance=min_distance,
                                            threshold_abs=threshold)

            delta_y = y_pred['offsets'][ind,0]
            delta_x = y_pred['offsets'][ind,1]

            dot_temp = np.zeros_like(dot_pixel_inds, dtype=np.float32)

            for i, (y_ind, x_ind) in enumerate(dot_pixel_inds):
                dot_temp[i, 0] = y_ind + delta_y[y_ind, x_ind] + y_offsets[ind]
                dot_temp[i, 1] = x_ind + delta_x[y_ind, x_ind] + x_offsets[ind]
            dot_centers.append(dot_temp)

        return np.vstack(dot_centers)
    
    def get_spot_intensities(
            self,
            transforms,
            threshold=0.9,
            min_distance=1,
            extra_pixel_num=0
    ):
        
        spot_probs = transforms['detections'][:, 1:2] # Shape R, C, H, W
        prob_mips = np.max(spot_probs, axis=0) # Shape C, H, W
        
        point_coord = max_cp_array_to_point_list_max(prob_mips, threshold=threshold, min_distance=min_distance)

        intensities = extract_spots_prob_from_coords_maxpool(spot_probs, point_coord, extra_pixel_num=extra_pixel_num)        
        intensities = np.concatenate(intensities, axis=1)

        return intensities, point_coord


    def _y_annotations_to_point_list_max(self, y_pred, threshold=0.95, min_distance=2):
        """Convert raw prediction to a predicted point list using
        ``skimage.feature.peak_local_max`` to determine local maxima in classification
        prediction image, and their corresponding regression values will be used to
        create a final spot position prediction which will be added to the output spot
        center coordinates list.

        Args:
            y_pred (array): a dictionary of predictions with keys `'classification'` and
                `'offset_regression'` corresponding to the named outputs of the
                ``dot_net_2D model``.
            threshold (float): a number in ``[0, 1]``. Pixels with classification
                score > `threshold` are considered as containing a spot center.
            min_distance (float): the minimum distance between detected spots in pixels.

        Returns:
            array: spot center coordinates of the format [[y0, x0], [y1, x1],...]
        """
        if not isinstance(y_pred, dict):
            raise TypeError('Input predictions must be a dictionary.')

        dot_centers = []

        for ind in range(np.shape(y_pred['detections'])[0]):
            # iterate through individual images in this batch

            dot_pixel_inds = peak_local_max(y_pred['detections'][ind, 1],
                                            min_distance=min_distance,
                                            threshold_abs=threshold)

            delta_y = y_pred['offsets'][ind,0]
            delta_x = y_pred['offsets'][ind,1]

            dot_temp = np.zeros_like(dot_pixel_inds, dtype=np.float32)

            for i, (y_ind, x_ind) in enumerate(dot_pixel_inds):
                dot_temp[i, 0] = y_ind + delta_y[y_ind, x_ind]
                dot_temp[i, 1] = x_ind + delta_x[y_ind, x_ind]
            dot_centers.append(dot_temp)

        return np.vstack(dot_centers)
    
    def predict_transforms(
            self,
            X,
            batch_size=20,
        ):
        
        # Preprocess
        X = spotnet_preprocess(X)
        tiles, tile_info = tile_input(X, (self.input_shape,self.input_shape))

        # Send data to device
        X = torch.tensor(tiles, device=self.device)

        batches = torch.split(X, batch_size)

        transforms = {
            'offsets': [],
            'detections': []
        }
        
        for batch in batches:

            # Infer
            with torch.inference_mode():
                pred = self.model(batch)
            

            for k,v in pred.items():
                transforms[k].append(v.cpu().numpy())

        for k,v in transforms.items():
            transforms[k] = np.concatenate(v)

        for k,v in transforms.items():
            transforms[k] = untile_output(v, tile_info)    

        return transforms

    
    def _eval_predict(
            self,
            X,
            threshold=0.99,
            min_distance=1
    ):
        
        """Input of X in shape (B, 1, H, W)
        threshold of finding local maxima
        min distance between the points predicted
        """

        # Preprocess
        X = spotnet_preprocess(X)

        # Send data to device
        X = torch.tensor(X, device=self.device)

        # Infer
        with torch.inference_mode():
            transforms = self.model(X)

        for k,v in transforms.items():
            transforms[k] = v.cpu().numpy()

        # Postprocess
        points = self._y_annotations_to_point_list_max(transforms, threshold=threshold, min_distance=min_distance)        

        return points
        

    def predict_points(
            self,
            X,
            threshold=0.99,
            min_distance=1,
            batch_size=20,
    ):
        
        """Input of X in shape (B, 1, H, W)
        threshold of finding local maxima
        min distance between the points predicted
        """

        # Preprocess
        X = spotnet_preprocess(X)
        tiles, tile_info = tile_input(X, (self.input_shape,self.input_shape))

        # Get the point offsets for each tile
        x_offset = np.array(tile_info['y_starts'])
        y_offset = np.array(tile_info['x_starts'])


        # Send data to device
        X = torch.tensor(tiles, device=self.device)

        batches = torch.split(X, batch_size)

        points = []
        counter = 0
        for batch in batches:

            # Infer
            with torch.inference_mode():
                transforms = self.model(batch)

            for k,v in transforms.items():
                transforms[k] = v.cpu().numpy()

            # Postprocess with offsets
            curr_x_offset = x_offset[counter:counter+batch.shape[0]]
            curr_y_offset = y_offset[counter:counter+batch.shape[0]]

            curr_points = self._y_annotations_to_point_list_max_offsets(
                transforms, threshold=threshold, min_distance=min_distance,
                x_offsets=curr_x_offset, y_offsets=curr_y_offset
            )
            points.append(curr_points)
            counter+=batch.shape[0]

        points = np.vstack(points)        

        return points
