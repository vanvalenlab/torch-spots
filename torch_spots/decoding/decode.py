import numpy as np

import numpy as np
import pandas as pd

from scipy.spatial import distance
from tqdm import tqdm

from .decoding_utils import decoding_function


class SpotDecoding:
    """Initialize a model for spot decoding of multiplex images.

    The ``predict`` method handles inference procedure. It infers the
    model parameters and predicts the spot identities.

    Example::

        from torch_spots.decoding.decode import SpotDecoding

        # Create the application
        app = SpotDecoding(df_barcodes, rounds, channels)

        # Decode the spots
        decoding_dict = app.predict(spot_probabilities)

    Args:
        df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
                Index:
                    RangeIndex (starting from 1)
                Columns:
                    Name: Gene, dtype: object
                    Name: r0c0, dtype: int64
                    Name: r0c1, dtype: int64
                    Name: r1c0, dtype: int64
                    Name: r1c1, dtype: int64
                    ...
                    Name: r9c0, dtype: int64
                    Name: r9c1, dtype: int64

        rounds (int): Number of rounds.
        channels (int): Number of channels.
        params_mode (str): Number of model parameters, whether the parameters are shared across
            channels or rounds for the Relaxed Bernoulli model.
            Valid options: `['2', '2*R', '2*C', '2*R*C']`. Defaults to `'2*R*C'`. 

    Raises:
        ValueError: `df_barcodes` must be a Pandas DataFrame.
        ValueError: The first column of `df_barcodes` must be called `'Gene'`. It should contain
            the names of the genes in the codebook.
        ValueError: The number of columns in `df_barcodes` must equal `rounds * channels + 1`.
        ValueError: Barcode values must be 0 or 1.
        ValueError: Codebook genes should not include `'Background'` or `'Unknown'`.
    """

    dataset_metadata = {}
    model_metadata = {}

    def __init__(self,
                 df_barcodes,
                 rounds,
                 channels,
                 params_mode='2*R*C'):
        self.rounds = rounds
        self.channels = channels
        self.params_mode = params_mode

        self._validate_codebook(df_barcodes)
        self.df_barcodes = self._add_bkg_unknown_to_barcodes(df_barcodes)

    def _validate_codebook(self, df_barcodes):
        """Validate the format of the input codebook.

        Args:
            df_barcodes (pandas.DataFrame): Codebook, the first column is gene names (`'Gene'`),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
                Index:
                    RangeIndex (starting from 1)
                Columns:
                    Name: Gene, dtype: object
                    Name: r0c0, dtype: int64
                    Name: r0c1, dtype: int64
                    Name: r1c0, dtype: int64
                    Name: r1c1, dtype: int64
                    ...
                    Name: r9c0, dtype: int64
                    Name: r9c1, dtype: int64
        """
        if not isinstance(df_barcodes, pd.DataFrame):
            raise TypeError('df_barcodes must be a Pandas DataFrame')
        
        if df_barcodes.columns[0] != 'Gene':
            raise ValueError('The first column of df_barcodes must contain the gene names and '
                             'have the column name \'Gene\'.')
        
        if len(df_barcodes.columns) != self.rounds * self.channels + 1:
            raise ValueError('The length of the barcode must equal rounds * channels.')
        
        valid_vals = {0,1}
        vals = df_barcodes.values[:, 1:]
        if valid_vals != set(vals.flatten()):
            raise ValueError('Barcode values must be 0 or 1.')

        if df_barcodes['Gene'].isin(['Background', 'Unknown']).any():
            raise ValueError('The Gene column should not include the reserved names '
                             '\'Background\' or \'Unknown\'. These classes are added '
                             'automatically.')

    def _add_bkg_unknown_to_barcodes(self, df_barcodes):
        """Add `Background` and `Unknown` barcodes to the codebook. The barcode of
        `Background` is all zeros and the barcode for `Unknown` is all -1s.

        Args:
            df_barcodes (pd.DataFrame): Codebook, the first column is gene names (`'Gene'`),
            the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
            For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
                Index:
                    RangeIndex (starting from 1)
                Columns:
                    Name: Gene, dtype: object
                    Name: r0c0, dtype: int64
                    Name: r0c1, dtype: int64
                    Name: r1c0, dtype: int64
                    Name: r1c1, dtype: int64
                    ...
                    Name: r9c0, dtype: int64
                    Name: r9c1, dtype: int64

        Returns:
            pd.DataFrame: The augmented codebook.

        """
        df_barcodes_aug = df_barcodes.copy()
        barcode_len = df_barcodes_aug.shape[1] - 1
        df_barcodes_aug.loc[len(df_barcodes_aug)+1] = ['Background'] + [0] * (barcode_len)
        df_barcodes_aug.loc[len(df_barcodes_aug)+1] = ['Unknown'] + [-1] * (barcode_len)
        return df_barcodes_aug
    
    def _validate_spot_probabilities(self, spot_probabilities):
        """Validate values of spot probabilities before spot decoding prediction.

        Args:
            spot_probabilities (numpy.array): Array of spot probability values with shape
                [num_spots, (rounds * channels)].

        Raises:
            ValueError: The number of columns must equal `rounds * channels`.
            ValueError: Spot probabilities must be in [0, 1].
        """
        expected = self.rounds * self.channels
        if spot_probabilities.ndim != 2 or spot_probabilities.shape[1] != expected:
            raise ValueError('Spot probabilities must have shape [num_spots, {}] (one column per '
                             'codebook bit), got shape {}.'.format(expected,
                                                                   tuple(spot_probabilities.shape)))
        if (spot_probabilities > 1).any() or (spot_probabilities < 0).any():
            raise ValueError('Spot probability values must be in [0, 1], got values from {} to {}.'
                             .format(spot_probabilities.min(), spot_probabilities.max()))

    def _decoding_output_to_dict(self, out):
        """Convert decoding output to dictionary.

        Args:
            out (dict): Dictionary with keys: `'class_probs'`, `'params'`.

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.

        """
        barcodes_idx2name = dict(
            zip(1 + np.arange(len(self.df_barcodes)), self.df_barcodes.Gene.values))
        decoded_dict = {}
        decoded_dict['probability'] = out['class_probs'].max(axis=1)
        decoded_dict['spot_index'] = np.arange(len(decoded_dict['probability']))
        decoded_dict['predicted_id'] = out['class_probs'].argmax(axis=1) + 1
        # object dtype: a fixed-width string array would cut 'Unknown' or a rescued gene name
        # down to the width of the longest name predicted so far
        decoded_dict['predicted_name'] = np.array(
            list(map(barcodes_idx2name.get, decoded_dict['predicted_id'])), dtype=object)
        decoded_dict['source'] = np.repeat(
            'prediction', len(decoded_dict['probability'])).astype('U25')
        return decoded_dict

    def _threshold_unknown_by_prob(self, decoded_dict, unknown_index, pred_prob_thresh=0.95):
        """Threshold the decoded spots to identify unknown. If the highest probability
        if below a certain threshold, the spot will be classfied as Unknown.

        Args:
            decoded_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
            unknown_index (int): The index for `Unknown` category.

        Returns:
            dict: similar to input, just replace the low probability
                ones with `Unknown`.
        """
        decoded_dict['predicted_id'][decoded_dict['probability'] < pred_prob_thresh] = unknown_index
        decoded_dict['predicted_name'][decoded_dict['probability'] < pred_prob_thresh] = 'Unknown'
        return decoded_dict

    def _rescue_errors(self,
                       decoding_dict,
                       spot_probabilities):
        """Rescues decoded spots assigned as `'Background'` or `'Unknown'` by if their spot
        probability values have a Hamming distance of 1 from each of the barcodes.

        Args:
            decoding_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`. This dictionary has already been processed to
                assign low probability predictions to `'Unknown'`.
            spot_probabilities (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
        """

        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        barcodes_array = self.df_barcodes[ch_names].values
        num_barcodes = barcodes_array.shape[0]
        barcode_len = barcodes_array.shape[1]

        predicted_ids = decoding_dict['predicted_id']
        predicted_names = decoding_dict['predicted_name']
        sources = decoding_dict['source']

        attempted = 0
        successful = 0
        for i,pred in tqdm(enumerate(predicted_names)):
            if pred in ['Background', 'Unknown']:
                attempted += 1
                dist_list = np.zeros(num_barcodes)
                for ii in range(num_barcodes):
                    dist_list[ii] = distance.hamming(np.round(spot_probabilities[i]),
                                                     barcodes_array[ii])
                scaled_dist_list = dist_list * barcode_len
                if 1 in scaled_dist_list:
                    successful += 1

                    new_gene = np.argwhere(scaled_dist_list == 1)[0][0]
                    # gene ids are 1-indexed
                    predicted_ids[i] = new_gene + 1
                    predicted_names[i] = self.df_barcodes['Gene'].values[new_gene]
                    sources[i] = 'error rescue'
        
        result = {
            'spot_index': decoding_dict['spot_index'],
            'predicted_id': predicted_ids,
            'predicted_name': predicted_names,
            'probability': decoding_dict['probability'],
            'source': sources
        }

        print('{} of {} rescue attempts were successful.'.format(successful, attempted))
        
        return(result)
    
    def _rescue_mixed_spots(self,
                            decoding_dict,
                            spot_probabilities,
                            prob_threshold=0.95):
        """Rescues decoded spots assigned as `'Background'` or `'Unknown'` by if their spot
        probability values have a Hamming distance of 1 from each of the barcodes.

        Args:
            decoding_dict (dict): Dictionary containing decoded spot identities with
                keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
            spot_probabilities (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`.
            df_barcodes (pd.DataFrame): Codebook, the first column is gene names (`'Gene'`),
                the rest are binary barcodes, encoded using 1 and 0. Index should start at 1.
                For exmaple, for a (rounds=10, channels=2) codebook, it should look like::
            
                Index:
                    RangeIndex (starting from 1)
                Columns:
                    Name: Gene, dtype: object
                    Name: r0c0, dtype: int64
                    Name: r0c1, dtype: int64
                    Name: r1c0, dtype: int64
                    Name: r1c1, dtype: int64
                    ...
                    Name: r9c0, dtype: int64
                    Name: r9c1, dtype: int64

        Returns:
            dict: Dictionary with keys: `'spot_index'`, `'probability'`, `'predicted_id'`,
                `'predicted_name'`, `'source'`.
        """

        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        barcodes_array = self.df_barcodes[ch_names].values
        num_barcodes = barcodes_array.shape[0]
        barcode_len = barcodes_array.shape[1]

        spot_indices = decoding_dict['spot_index']
        predicted_ids = decoding_dict['predicted_id']
        predicted_names = decoding_dict['predicted_name']
        probabilities = decoding_dict['probability']
        sources = decoding_dict['source']

        attempted = 0
        successful = 0
        for i,prob in tqdm(enumerate(probabilities)):
            if prob < prob_threshold:
                attempted += 1
                gene_id = predicted_ids[i]
                if gene_id > num_barcodes-2:
                    continue
                
                # gene ids are 1-indexed
                barcode = barcodes_array[gene_id-1]
                probabilities_updated = spot_probabilities[i].copy()
                probabilities_updated[barcode==1] = 0

                dist_list = np.zeros(num_barcodes)
                for ii in range(num_barcodes):
                    dist_list[ii] = distance.hamming(np.round(probabilities_updated),
                                                     barcodes_array[ii])
                scaled_dist_list = dist_list * barcode_len
                if 1 in scaled_dist_list:
                    successful += 1
                    spot_indices = np.append(spot_indices, [spot_indices[i]])

                    new_id = np.argwhere(scaled_dist_list == 1)[0][0]
                    # gene ids are 1-indexed
                    predicted_ids = np.append(predicted_ids, [new_id + 1])

                    new_name = self.df_barcodes['Gene'].values[new_id]
                    predicted_names = np.append(predicted_names, [new_name])

                    probabilities = np.append(probabilities, [-1])

                    sources = np.append(sources, 'mixed rescue')

        result = {
            'spot_index': spot_indices,
            'predicted_id': predicted_ids,
            'predicted_name': predicted_names,
            'probability': probabilities,
            'source': sources
        }

        print('{} of {} rescue attempts were successful.'.format(successful, attempted))

        return(result)

    def predict(self,
                spot_probabilities,
                locations=None,
                *,
                num_iter=500,
                batch_size=1000,
                seed=1,
                pred_prob_thresh=0.95,
                rescue_errors=True,
                rescue_mixed=False,
                genes_only=False):
        """Predict the gene assignment of each spot.

        Args:
            spot_probabilities (numpy.array): Array of spot probability values with shape
                `[num_spots, (rounds * channels)]`, in [0, 1]. Column *i* belongs to codebook bit
                column *i*.
            locations (numpy.array): Optional integer `[y, x]` (row, column) location of each spot,
                shape `[num_spots, 2]`, as returned by the detection hand-off. When given, the
                table has `y` and `x` columns.
            num_iter (int): Number of iterations for training. Defaults to 500.
            batch_size (int): Size of batches for training. Defaults to 1000.
            seed (int): Seed for randomness. It sets which minibatch of spots each training
                iteration sees. Defaults to 1.
            pred_prob_thresh (float): The threshold of unknown category, within [0,1]. Defaults to 0.95.
            rescue_errors (bool): Whether to check if `'Background'`-  and `'Unknown'`-assigned
                spots have a Hamming distance of 1 to other barcodes.
            rescue_mixed (bool): Whether to check if low probability predictions are the result of
                two mixed barcodes.
            genes_only (bool): Drop every row labelled `'Background'`, `'Unknown'` or `'Blank'`
                (a name containing `Blank`). Applied last, so it never changes which spots are
                rescued. Defaults to False.

        Returns:
            pandas.DataFrame: One row per spot, in input order, followed by any rows added by
                mixed rescue. Columns: `'spot_index'`, `'y'`, `'x'` (only with `locations`),
                `'predicted_id'`, `'predicted_name'`, `'probability'`, `'source'`, then one
                `'prob_<codebook column>'` per bit. The settings and the fitted weights, sigma,
                temperature and per-iteration losses are kept on `self.last_run`.
        """

        spot_probabilities = np.asarray(spot_probabilities)
        self._validate_spot_probabilities(spot_probabilities)
        if locations is not None:
            locations = np.asarray(locations)
            if locations.shape != (spot_probabilities.shape[0], 2):
                raise ValueError('locations must have shape [num_spots, 2] = {}, got {}.'.format(
                    (spot_probabilities.shape[0], 2), tuple(locations.shape)))

        spot_probabilities_reshaped = np.reshape(spot_probabilities,
                                                (-1, self.channels, self.rounds))

        # convert df_barcodes to an array
        ch_names = list(self.df_barcodes.columns)
        ch_names.remove('Gene')
        unknown_index = self.df_barcodes.index.max()
        barcodes_array = self.df_barcodes[ch_names].values.reshape(-1, self.channels,
                                                                   self.rounds)[:-1, :, :]

        # decode
        out = decoding_function(spot_probabilities_reshaped,
                                barcodes_array,
                                num_iter=num_iter,
                                batch_size=batch_size,
                                seed=seed,
                                params_mode=self.params_mode)
        decoding_dict = self._decoding_output_to_dict(out)
        decoding_dict_trunc = self._threshold_unknown_by_prob(
            decoding_dict, unknown_index, pred_prob_thresh=pred_prob_thresh)

        if rescue_errors:
            print('Revising errors...')
            decoding_dict_trunc = self._rescue_errors(decoding_dict_trunc,
                                                      spot_probabilities)
        if rescue_mixed:
            print('Correcting mixed barcodes...')
            decoding_dict_trunc = self._rescue_mixed_spots(decoding_dict_trunc,
                                                           spot_probabilities)

        table = self._build_table(decoding_dict_trunc, spot_probabilities, locations, ch_names)
        if genes_only:
            names = table['predicted_name']
            table = table[~(names.isin(['Background', 'Unknown'])
                            | names.str.contains('Blank', regex=False))].reset_index(drop=True)

        params = out.get('params', {})
        self.last_run = {
            'settings': {'params_mode': self.params_mode, 'num_iter': num_iter,
                         'batch_size': batch_size, 'seed': seed,
                         'pred_prob_thresh': pred_prob_thresh, 'rescue_errors': rescue_errors,
                         'rescue_mixed': rescue_mixed, 'genes_only': genes_only},
            'weights': _to_numpy(params.get('w_star')),
            'sigma': _to_numpy(params.get('sigma_star')),
            'temperature': _to_numpy(params.get('temperature_star')),
            'losses': list(params.get('losses', [])),
        }

        return table

    @staticmethod
    def _build_table(decoding_dict, spot_probabilities, locations, bit_names):
        """Assemble the output table. `y`, `x` and the probabilities are looked up through
        `spot_index`, so rows added by mixed rescue carry their own spot's values."""
        spot_index = np.asarray(decoding_dict['spot_index'])
        columns = {'spot_index': spot_index}
        if locations is not None:
            columns['y'] = locations[spot_index, 0]
            columns['x'] = locations[spot_index, 1]
        columns['predicted_id'] = np.asarray(decoding_dict['predicted_id'])
        columns['predicted_name'] = np.asarray(decoding_dict['predicted_name'])
        columns['probability'] = np.asarray(decoding_dict['probability'])
        columns['source'] = np.asarray(decoding_dict['source'])
        table = pd.DataFrame(columns)
        probs = pd.DataFrame(spot_probabilities[spot_index],
                             columns=['prob_{}'.format(name) for name in bit_names])
        return pd.concat([table, probs], axis=1)


def _to_numpy(value):
    return None if value is None else value.detach().cpu().numpy() if hasattr(value, 'detach') \
        else np.asarray(value)
