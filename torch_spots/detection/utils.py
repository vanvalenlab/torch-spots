"""Variational Inference functions for spot decoding. Code adapted from PoSTcode
https://github.com/gerstung-lab/postcode (https://doi.org/10.1101/2021.10.12.464086)."""

import numpy as np

from skimage import measure
from skimage.feature import peak_local_max
from scipy.signal import windows

########################################################################
#                      Postprocess functions                           #
########################################################################

def y_annotations_to_point_list(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    pixel as containing dot > `threshold`, , and their corresponding regression
    values will be used to create a final spot position prediction which will
    be added to the output spot center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        ind (int): the index of the image in the batch for which to convert the
            annotations.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')

    dot_centers = []
    for ind in range(np.shape(y_pred['detections'])[0]):
        contains_dot = y_pred['detections'][ind, ..., 1] > threshold
        delta_y = y_pred['offsets'][ind, ..., 0]
        delta_x = y_pred['offsets'][ind, ..., 1]

        dot_pixel_inds = np.argwhere(contains_dot)
        dot_centers.append([[y_ind + delta_y[y_ind, x_ind], x_ind +
                             delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds])

    return np.array(dot_centers)


def y_annotations_to_point_list_restrictive(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    pixel as containing dot > `threshold` AND center regression is contained
    in the pixel. The corresponding regression values will be used to create
    a final spot position prediction which will be added to the output spot
    center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        ind (int): the index of the image in the batch for which to convert the
            annotations.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``.
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')

    dot_centers = []
    for ind in range(np.shape(y_pred['classification'])[0]):
        contains_dot = y_pred['classification'][ind, 1] > threshold
        delta_y = y_pred['offset_regression'][ind, 0]
        delta_x = y_pred['offset_regression'][ind, 1]
        contains_its_regression = (abs(delta_x) <= 0.5) & (abs(delta_y) <= 0.5)

        final_dot_detection = contains_dot & contains_its_regression

        dot_pixel_inds = np.argwhere(final_dot_detection)
        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
              x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return np.array(dot_centers)


def y_annotations_to_point_list_max(y_pred, threshold=0.95, min_distance=2):
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
        dot_pixel_inds = peak_local_max(y_pred['detections'][ind, 1],
                                        min_distance=min_distance,
                                        threshold_abs=threshold)

        delta_y = y_pred['offsets'][ind,0]
        delta_x = y_pred['offsets'][ind,1]

        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
              x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return np.array(dot_centers)


def max_cp_array_to_point_list_max(max_cp_array, threshold=0.95, min_distance=2):
    """Convert raw prediction to a predicted point list using
    ``skimage.feature.peak_local_max`` to determine local maxima in classification
    prediction image, and their corresponding regression values will be used to
    create a final spot position prediction which will be added to the output spot
    center coordinates list. This is performed on the max projected cp_array.

    Args:
        max_cp_array (array): (batch, x, y)
        threshold (float): A number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered as containing a spot center.
        min_distance (float): The minimum distance between detected spots in pixels.

    Returns:
        array: Spot center coordinates, list (num_images,) with each entry shape
        (num_spots, 2).
    """
    dot_centers = []
    for ind in range(np.shape(max_cp_array)[0]):
        dot_pixel_inds = peak_local_max(max_cp_array[ind, ...],
                                        min_distance=min_distance,
                                        threshold_abs=threshold)
        dot_centers.append(dot_pixel_inds)

    return np.array(dot_centers)


def y_annotations_to_point_list_cc(y_pred, threshold=0.95):
    """Convert raw prediction to a predicted point list: classification of
    connected component as containing dot > `threshold`, , and their corresponding
    regression values will be used to create a final spot position prediction which
    will be added to the output spot center coordinates list.

    Args:
        y_pred (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` corresponding to the named outputs of the
            ``dot_net_2D model``.
        threshold (float): a number in ``[0, 1]``. Pixels with classification
            score > `threshold` are considered containing a spot center.

    Returns:
        array: spot center coordinates of the format ``[[y0, x0], [y1, x1],...]``
    """
    if not isinstance(y_pred, dict):
        raise TypeError('Input predictions must be a dictionary.')
    if 'classification' not in y_pred.keys() or 'offset_regression' not in y_pred.keys():
        raise NameError('Input must have keys \'classification\' and \'offset_regression\'')

    dot_centers = []
    for ind in range(np.shape(y_pred['classification'])[0]):

        delta_y = y_pred['offset_regression'][ind, 0]
        delta_x = y_pred['offset_regression'][ind, 1]

        blobs = y_pred['classification'][ind, 1] > threshold
        label_image = measure.label(blobs, background=0)
        rp = measure.regionprops(label_image)

        dot_centers_temp = []
        for region in rp:
            region_pixel_inds = region.coords
            reg_pred = [[y_ind + delta_y[y_ind, x_ind], x_ind + delta_x[y_ind, x_ind]]
                        for y_ind, x_ind in region_pixel_inds]
            dot_centers_temp.append(np.mean(reg_pred, axis=0))

        dot_centers.append(dot_centers_temp)
    return np.concatenate(dot_centers)

def extract_spots_prob_from_coords_maxpool(image, spots_locations, extra_pixel_num=1):
    """Perform a max pooling and extract the intensities for each spot.

    Args:
        image (numpy.array): Probability maps with shape ``[batch, x, y, channel]``.
        spots_locations (numpy.array): Coordiantes found by max projection, used as anchor
            points for further max pooling operation. Shape ``[num_spots, 2]``.
        extra_pixel_num (int): Parameter for size of the pool. Defaults to 1, meaning
            a pool with size=(1,0,-1)x(1,0,-1)

    Returns:
        list: Spots intensities, each entry in the list is a numpy.array with shape
        ``[num_spots, channel]``.
    """

    if extra_pixel_num < 0:
        raise ValueError(
            "extra_pixel_num should be a positive integer \
            (zero included), but got {}".format(
                extra_pixel_num
            )
        )

    spots_intensities = []
    for idx_batch in range(len(image)):
        image_slice = image[idx_batch]
        coords = spots_locations[idx_batch]

        num_spots = len(coords)
        img_boundary_x = image_slice.shape[0] - 1
        img_boundary_y = image_slice.shape[1] - 1

        intensity_d = np.zeros(((extra_pixel_num * 2 + 1) ** 2, num_spots, image_slice.shape[-1]))
        d = -1
        for dx in np.arange(-extra_pixel_num, extra_pixel_num + 1):
            for dy in np.arange(-extra_pixel_num, extra_pixel_num + 1):
                d = d + 1
                for ind_cr in range(image_slice.shape[-1]):
                    x_coord = np.maximum(
                        0, np.minimum(img_boundary_x, np.around(coords[:, 0]) + dx)
                    )  # (num_spots,)
                    y_coord = np.maximum(
                        0, np.minimum(img_boundary_y, np.around(coords[:, 1]) + dy)
                    )  # (num_spots,)

                    intensity_d[d, :, ind_cr] = image_slice[x_coord, y_coord, ind_cr]

        intensity = np.max(intensity_d, axis=0)
        spots_intensities.append(intensity)

    return spots_intensities


########################################################################
#                          Image Functions                             #
########################################################################

def spline_window(window_size, overlap_left, overlap_right, power=2):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """

    def _spline_window(w_size):
        intersection = int(w_size / 4)
        wind_outer = (abs(2 * (windows.triang(w_size))) ** power) / 2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2 * (windows.triang(w_size) - 1)) ** power) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.amax(wind)
        return wind

    # Create the window for the left overlap
    if overlap_left > 0:
        window_size_l = 2 * overlap_left
        l_spline = _spline_window(window_size_l)[0:overlap_left]

    # Create the window for the right overlap
    if overlap_right > 0:
        window_size_r = 2 * overlap_right
        r_spline = _spline_window(window_size_r)[overlap_right:]

    # Put the two together
    window = np.ones((window_size,))
    if overlap_left > 0:
        window[0:overlap_left] = l_spline
    if overlap_right > 0:
        window[-overlap_right:] = r_spline

    return window

def window_2D(window_size, overlap_x=(32, 32), overlap_y=(32, 32), power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Returns a channels-first compatible window of shape (1, tile_x, tile_y).
    """
    window_x = spline_window(window_size[0], overlap_x[0], overlap_x[1], power=power)
    window_y = spline_window(window_size[1], overlap_y[0], overlap_y[1], power=power)

    # Reshape for channels-first broadcasting: (tile_x, 1) * (1, tile_y) -> (tile_x, tile_y)
    window = window_x[:, np.newaxis] * window_y[np.newaxis, :]

    # Add channel dim at front: (1, tile_x, tile_y) for broadcasting over (C, tile_x, tile_y)
    return window[np.newaxis, :, :]


def untile_image(tiles, tiles_info, power=2, **kwargs):
    """Untile a set of tiled images back to the original model shape.
    Expects channels-first format: (B, C, H, W).

    Args:
        tiles (numpy.array): The tiled images to untile, shape (N, C, tile_x, tile_y).
        tiles_info (dict): Details of how the image was tiled (from tile_image).
        power (int): The power of the window function.

    Returns:
        numpy.array: The untiled image, shape (B, C, H, W).
    """
    min_tile_size = 32
    min_stride_ratio = 0.5

    stride_ratio = tiles_info['stride_ratio']
    image_shape = tiles_info['image_shape']
    tile_size_x = tiles_info['tile_size_x']
    tile_size_y = tiles_info['tile_size_y']
    x_pad = tiles_info['pad_x']
    y_pad = tiles_info['pad_y']

    # Channels-first: (B, C, H, W) — use tiles.shape[1] for n_channels
    image_shape = (image_shape[0], tiles.shape[1], image_shape[2], image_shape[3])
    image = np.zeros(image_shape, dtype=float)

    # Precompute window cache keyed on overlap pairs
    window_cache = {}
    for overlap_x, overlap_y in zip(tiles_info['overlaps_x'], tiles_info['overlaps_y']):
        key = (overlap_x, overlap_y)
        if key not in window_cache:
            window_cache[key] = window_2D(
                (tile_size_x, tile_size_y),
                overlap_x=overlap_x,
                overlap_y=overlap_y,
                power=power
            )

    use_spline = (
        min_tile_size <= tile_size_x < image_shape[2] and
        min_tile_size <= tile_size_y < image_shape[3] and
        stride_ratio >= min_stride_ratio
    )

    for tile, batch, x_start, x_end, y_start, y_end, overlap_x, overlap_y in zip(
            tiles,
            tiles_info['batches'],
            tiles_info['x_starts'], tiles_info['x_ends'],
            tiles_info['y_starts'], tiles_info['y_ends'],
            tiles_info['overlaps_x'], tiles_info['overlaps_y']):

        if use_spline:
            window = window_cache[(overlap_x, overlap_y)]
            # Channels-first slice: (C, tile_x, tile_y) * (1, tile_x, tile_y)
            image[batch, :, x_start:x_end, y_start:y_end] += tile * window
        else:
            image[batch, :, x_start:x_end, y_start:y_end] = tile

    image = image.astype(tiles.dtype)

    # Unpad spatial dims (axes 2 and 3)
    x_end = image_shape[2] - x_pad[1] if x_pad[1] != 0 else None
    y_end = image_shape[3] - y_pad[1] if y_pad[1] != 0 else None

    return image[:, :, x_pad[0]:x_end, y_pad[0]:y_end]


def tile_image(image, model_input_shape=(512, 512), stride_ratio=1.0, pad_mode='constant'):
    """
    Tile large image into overlapping tiles of size `model_input_shape`.
    Expects and returns channels-first format: (B, C, H, W).

    Args:
        image (numpy.array): The image to tile, must be rank 4 (B, C, H, W).
        model_input_shape (tuple): The (x, y) spatial input size of the model.
        stride_ratio (float): Stride as a fraction of tile size.
        pad_mode (str): Padding mode passed to ``np.pad``.

    Returns:
        tuple: (numpy.array, dict): Tiled images array (B, C, tile_x, tile_y)
            and tiling metadata dict.

    Raises:
        ValueError: image is not rank 4.
    """
    if image.ndim != 4:
        raise ValueError('Expected image of rank 4, got {}'.format(image.ndim))

    tile_size_x, tile_size_y = model_input_shape
    # Channels-first: spatial dims are axes 2 and 3
    image_size_x, image_size_y = image.shape[2], image.shape[3]

    def even_stride(ratio, tile_size):
        return min(int(np.ceil(ratio * tile_size / 2.0) * 2), tile_size)

    stride_x = even_stride(stride_ratio, tile_size_x)
    stride_y = even_stride(stride_ratio, tile_size_y)

    rep_x = max(int(np.ceil((image_size_x - tile_size_x) / stride_x + 1)), 1)
    rep_y = max(int(np.ceil((image_size_y - tile_size_y) / stride_y + 1)), 1)

    def edge_padding(overlap):
        return (int(np.ceil(overlap / 2)), int(np.floor(overlap / 2)))

    overlap_x = tile_size_x + stride_x * (rep_x - 1) - image_size_x
    overlap_y = tile_size_y + stride_y * (rep_y - 1) - image_size_y
    pad_x = edge_padding(overlap_x)
    pad_y = edge_padding(overlap_y)

    # Channels-first padding: (B, C, H, W) → pad axes 2 and 3, not 1 and 2
    image = np.pad(image, [(0, 0), (0, 0), pad_x, pad_y], pad_mode)
    img_x, img_y = image.shape[2], image.shape[3]

    def tile_indices(rep, stride, tile_size, img_size):
        starts = [i * stride if i < rep - 1 else img_size - tile_size for i in range(rep)]
        ends = [s + tile_size for s in starts]
        return starts, ends

    x_starts, x_ends = tile_indices(rep_x, stride_x, tile_size_x, img_x)
    y_starts, y_ends = tile_indices(rep_y, stride_y, tile_size_y, img_y)

    def compute_overlaps(rep, stride, tile_size, starts, img_size):
        overlaps = []
        for i in range(rep):
            if i == 0:
                ov = (0, tile_size - stride)
            elif i == rep - 2:
                ov = (tile_size - stride, tile_size - img_size + starts[i] + tile_size)
            elif i == rep - 1:
                ov = (starts[i - 1] + tile_size - starts[i], 0) if rep > 1 else (0, 0)
            else:
                ov = (tile_size - stride, tile_size - stride)
            overlaps.append(ov)
        return overlaps

    overlaps_x = compute_overlaps(rep_x, stride_x, tile_size_x, x_starts, img_x)
    overlaps_y = compute_overlaps(rep_y, stride_y, tile_size_y, y_starts, img_y)

    batch_size, n_channels = image.shape[0], image.shape[1]
    n_tiles = batch_size * rep_x * rep_y

    # Channels-first tile shape: (N, C, tile_x, tile_y)
    tiles = np.zeros((n_tiles, n_channels, tile_size_x, tile_size_y), dtype=image.dtype)

    indices = [(b, i, j)
               for b in range(batch_size)
               for i in range(rep_x)
               for j in range(rep_y)]

    for counter, (b, i, j) in enumerate(indices):
        # Slice spatial dims 2 and 3, keeping all channels (dim 1)
        tiles[counter] = image[b, :, x_starts[i]:x_ends[i], y_starts[j]:y_ends[j]]

    flat_b, flat_i, flat_j = zip(*indices)

    return tiles, {
        'batches':      list(flat_b),
        'x_starts':     [x_starts[i] for i in flat_i],
        'x_ends':       [x_ends[i]   for i in flat_i],
        'y_starts':     [y_starts[j] for j in flat_j],
        'y_ends':       [y_ends[j]   for j in flat_j],
        'overlaps_x':   [overlaps_x[i] for i in flat_i],
        'overlaps_y':   [overlaps_y[j] for j in flat_j],
        'stride_x':     stride_x,
        'stride_y':     stride_y,
        'tile_size_x':  tile_size_x,
        'tile_size_y':  tile_size_y,
        'stride_ratio': stride_ratio,
        'image_shape':  image.shape,
        'dtype':        image.dtype,
        'pad_x':        pad_x,
        'pad_y':        pad_y,
    }

def percentile_threshold(image: np.typing.ArrayLike, percentile=99.9):
    """Threshold an image to reduce bright spots

    Args:
        image: numpy array of image data with expected shape `[batch, C, H, W]`
        percentile: cutoff used to threshold image

    Returns:
        np.array: thresholded version of input image
    """
    processed_image = np.zeros_like(image, dtype=np.float32)

    B, C, _, _ = image.shape
    flat = image.reshape(B, C, -1)  # [B, C, H*W]

    has_nonzero = (flat > 0).any(axis=-1)  # [B, C]

    # Compute percentile over spatial dims; mask zero values with nan to exclude them
    flat_nan = np.where(flat > 0, flat.astype(float), np.nan)
    thresholds = np.nanpercentile(flat_nan, percentile, axis=-1)  # [B, C]

    # Only apply threshold where channel isn't blank
    thresholds = np.where(has_nonzero, thresholds, np.nan)  # [B, C]

    # Clip each (img, channel) slice to its threshold — expand dims for broadcasting
    thresh_expanded = thresholds[:, :, np.newaxis, np.newaxis]  # [B, C, 1, 1]
    clipped = np.minimum(image, thresh_expanded)

    # Zero out blank channels (preserve zeros_like behavior)
    processed_image = np.where(has_nonzero[:, :, np.newaxis, np.newaxis], clipped, processed_image)

    return processed_image.astype(np.float32)

def minmax_normalization(image):

    channel_min = np.min(image, axis=(1,2,3), keepdims=True).astype(np.float32)
    channel_max = np.max(image, axis=(1,2,3), keepdims=True).astype(np.float32)

    image_minmax = (image - channel_min)/(channel_max-channel_min)

    return image_minmax

# pre- and post-processing functions
def spotnet_preprocess(image):
    """Preprocess input data for Mesmer model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """

    if len(image.shape) != 4:
        raise ValueError(f"Image data must be 4D, got image of shape {image.shape}")

    output = np.copy(image)

    output = percentile_threshold(image=output, percentile=99.9)
    output = minmax_normalization(image=output)
    
    return output

def tile_input(image, model_image_shape, pad_mode='constant'):
    """
    Tile the input image to match shape expected by model.
    Expects channels-first format: (B, C, H, W).

    Args:
        image (numpy.array): Input image to tile, must be rank 4 (B, C, H, W).
        model_image_shape (tuple): The (H, W) spatial input size of the model.
        pad_mode (str): The padding mode, one of "constant" or "reflect".

    Returns:
        (numpy.array, dict): Tuple of tiled image and dict of tiling information.

    Raises:
        ValueError: Input images must have only 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(
            'tile_image only supports 4D images. '
            f'Image submitted has {image.ndim} dimensions.'
        )

    # Channels-first: spatial dims are axes 2 and 3
    x_diff = image.shape[2] - model_image_shape[0]
    y_diff = image.shape[3] - model_image_shape[1]

    if x_diff < 0 or y_diff < 0:
        # Pad spatial dims only, leave batch and channel dims untouched
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff // 2 + x_diff % 2)
        y_pad = (y_diff // 2, y_diff // 2 + y_diff % 2)

        tiles = np.pad(image, [(0, 0), (0, 0), x_pad, y_pad], 'reflect')
        tiles_info = {'padding': True, 'x_pad': x_pad, 'y_pad': y_pad}
    else:
        tiles, tiles_info = tile_image(
            image,
            model_input_shape=model_image_shape,
            stride_ratio=1.0,
            pad_mode=pad_mode
        )

    return tiles, tiles_info

def untile_output(output_tiles, tiles_info):
    """Untiles either a single array or a list of arrays.
    Expects channels-first format: (B, C, H, W).

    Args:
        output_tiles (numpy.array or list): Array or list of arrays.
        tiles_info (dict): Tiling specs output by the tiling function.
        model_image_shape (tuple): The (H, W) spatial input size of the model.

    Returns:
        numpy.array or list: Untiled image(s) in channels-first format.
    """
    if tiles_info.get('padding', False):
        def _process(im, tiles_info):
            (xl, xh), (yl, yh) = tiles_info['x_pad'], tiles_info['y_pad']
            # Channels-first: spatial dims are axes 2 and 3
            xh = -xh if xh != 0 else None
            yh = -yh if yh != 0 else None
            return im[:, :, xl:xh, yl:yh]
    else:
        def _process(im, tiles_info):
            return untile_image(im, tiles_info)

    if isinstance(output_tiles, list):
        return [_process(o, tiles_info) for o in output_tiles]
    return _process(output_tiles, tiles_info)
