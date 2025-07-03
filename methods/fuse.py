import numpy as np
from sklearn.decomposition import PCA

def concat(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:

    assert arr1.dtype == np.uint8 and arr2.dtype == np.uint8
    assert arr1.shape == arr2.shape, "Both arrays must have the same shape"

    return np.hstack((arr1, arr2))


def unconcat(packed_arr: np.ndarray) -> tuple:

    assert packed_arr.dtype == np.uint8
    try:
        h, w2 = packed_arr.shape
    except ValueError as ve:
        h, w2, c = packed_arr.shape
    w = w2 // 2

    original_arr1 = packed_arr[:, :w]
    original_arr2 = packed_arr[:, w:]

    return original_arr1, original_arr2

def fuse_pca(image_uint8):

    H, W, C = image_uint8.shape
    assert C == 6 and image_uint8.dtype == np.uint8

    # Convert to float32 for PCA
    image = image_uint8.astype(np.float32)
    pixels = image.reshape(-1, 6)

    # Standardize
    orig_mean = pixels.mean(axis=0)
    orig_std = pixels.std(axis=0)
    pixels_std = (pixels - orig_mean) / (orig_std + 1e-8)

    # PCA
    pca = PCA(n_components=6)
    transformed = pca.fit_transform(pixels_std)

    # Sort components by explained variance (ascending)
    variance_order = np.argsort(pca.explained_variance_)
    transformed_sorted = transformed[:, variance_order]

    # Normalize to [0, 255]
    min_val = transformed_sorted.min(axis=0)
    max_val = transformed_sorted.max(axis=0)
    normed = (transformed_sorted - min_val) / (max_val - min_val + 1e-8)
    reduced_uint8 = (normed * 255).astype(np.uint8).reshape(H, W, 6)

    # Interleave
    alligned_high = reduced_uint8[:, :, :3]
    alligned_low = reduced_uint8[:, :, 3:]
    packed = np.concatenate([alligned_high, alligned_low], axis=1)  # Interleave along width

    # Store transform info
    pca_info = {
        'pca': pca,
        'orig_mean': orig_mean,
        'orig_std': orig_std,
        'min_val': min_val,
        'max_val': max_val,
        'variance_order': variance_order,
        'original_width': W
    }

    return packed, pca_info

def unfuse_pca(packed_decompressed, pca_info, width=512):

    H, W2, C = packed_decompressed.shape
    assert C == 3 and packed_decompressed.dtype == np.uint8

    # Deinterleave
    alligned_high = packed_decompressed[:, :width, :]
    alligned_low = packed_decompressed[:, width:, :]
    alligned = np.concatenate([alligned_high, alligned_low], axis=2)

    reduced_uint8 = alligned.astype(np.float32) / 255.0
    reduced_pixels = reduced_uint8.reshape(-1, 6)

    # Undo normalization
    min_val = pca_info['min_val']
    max_val = pca_info['max_val']
    rescaled = reduced_pixels * (max_val - min_val + 1e-8) + min_val

    # Reorder to original PCA order
    variance_order = pca_info['variance_order']
    inverse_order = np.argsort(variance_order)
    reordered = rescaled[:, inverse_order]

    # Inverse PCA transform
    pca = pca_info['pca']
    restored_std = pca.inverse_transform(reordered)

    # De-standardize
    orig_mean = pca_info['orig_mean']
    orig_std = pca_info['orig_std']
    restored = restored_std * (orig_std + 1e-8) + orig_mean

    return np.clip(restored, 0, 255).astype(np.uint8).reshape(H, width, 6)