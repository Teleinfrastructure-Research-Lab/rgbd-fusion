import numpy as np
from PIL import Image, features
import io
import cv2
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr

SHUFFLE_PATTERN = [26, 5, 10, 15, 25, 11, 22, 6, 19, 12, 16, 9, 28, 14, 24, 20, 30, 1, 13, 18, 2, 17, 21, 3, 29, 4, 27, 31, 8, 23, 0, 7]

GLOBAL_DEPTH_CLIP_NEAR = 0.01
GLOBAL_DEPTH_CLIP_FAR = 1.0

def float32_to_uint8_uniform(value: np.float32, shuffle_pattern=SHUFFLE_PATTERN) -> np.ndarray:
    """
    Converts a float32 value to a shuffled 4x uint8 array using a given bit shuffle pattern.

    Parameters:
    - value: A single np.float32 value.
    - shuffle_pattern: A list of 32 integers representing a permutation of bit indices [0-31].

    Returns:
    - A np.ndarray of 4 uint8 values with bits shuffled according to the pattern.
    """
    if not isinstance(value, np.float32):
        value = np.float32(value)

    bytes_raw = value.tobytes()
    bit_str = ''.join(f'{byte:08b}' for byte in bytes_raw)

    shuffled_bits = ''.join(bit_str[i] for i in shuffle_pattern)
    shuffled_bytes = [int(shuffled_bits[i*8:(i+1)*8], 2) for i in range(4)]

    return np.array(shuffled_bytes, dtype=np.uint8)

def uint8_uniform_to_float32(shuffled_bytes: np.ndarray, shuffle_pattern=SHUFFLE_PATTERN) -> np.float32:
    """
    Reconstructs a float32 value from 4 shuffled uint8s using the inverse of the provided bit shuffle pattern.

    Parameters:
    - shuffled_bytes: A np.ndarray of 4 uint8 values.
    - shuffle_pattern: The original shuffle pattern used during encoding.

    Returns:
    - A reconstructed np.float32 value.
    """
    if shuffled_bytes.shape != (4,) or shuffled_bytes.dtype != np.uint8:
        raise ValueError("shuffled_bytes must be a np.ndarray of shape (4,) and dtype uint8")

    shuffled_bit_str = ''.join(f'{b:08b}' for b in shuffled_bytes)

    inverse_pattern = [0] * 32
    for i, idx in enumerate(shuffle_pattern):
        inverse_pattern[idx] = i

    original_bits = ''.join(shuffled_bit_str[inverse_pattern[i]] for i in range(32))
    original_bytes = int(original_bits, 2).to_bytes(4, byteorder='big')

    return np.frombuffer(original_bytes, dtype=np.float32,)[0]

def float32_to_uint8_mean_std_per_channel(array):
    """
    Normalize a float32 image (H, W, 3) to uint8 using mean-std normalization per channel.

    Returns:
        - uint8_array: values scaled to [0, 255]
        - means: per-channel means (shape: (3,))
        - stds: per-channel stds (shape: (3,))
    """
    assert array.ndim == 3 and array.shape[2] == 3, "Expected shape (H, W, 3)"

    uint8_array = np.zeros_like(array, dtype=np.uint8)
    means = np.zeros(3, dtype=np.float32)
    stds = np.zeros(3, dtype=np.float32)

    for c in range(3):
        channel = array[..., c]
        mean = channel.mean()
        std = channel.std()
        means[c] = mean
        stds[c] = std

        # Normalize to zero mean, unit variance, then scale to [0, 1] (approx)
        norm = (channel - mean) / (std + 1e-8)
        norm = np.clip(norm, -3, 3)  # Clip to ±3 std dev
        norm_scaled = (norm + 3) / 6  # Map [-3, 3] → [0, 1]
        uint8_array[..., c] = (norm_scaled * 255).astype(np.uint8)

    return uint8_array, means, stds

def float32_to_uniform_uint8s_numeric(value: float) -> np.ndarray:
    # Assuming `value` is already normalized between [0,1]
    int_val = np.uint32(np.round(value * 0xFFFFFFFF))

    bytes_out = np.array([
        (int_val >> 24) & 0xFF,
        (int_val >> 16) & 0xFF,
        (int_val >> 8) & 0xFF,
        int_val & 0xFF
    ], dtype=np.uint8)

    return bytes_out

def uniform_uint8s_numeric_to_float32(bytes_in: np.ndarray) -> float:
    int_val = (np.uint32(bytes_in[0]) << 24) | \
              (np.uint32(bytes_in[1]) << 16) | \
              (np.uint32(bytes_in[2]) << 8)  | \
              np.uint32(bytes_in[3])

    value = int_val / 0xFFFFFFFF

    return np.float32(value)

def float32_array_to_uniform_uint8s(arr_float32: np.ndarray) -> np.ndarray:
    """Converts an entire float32 array into a uint8 array with shape (*arr_float32.shape, 4)."""
    assert arr_float32.dtype == np.float32
    
    # Convert float32 array into uint32 integers
    ints = np.round(arr_float32 * 0xFFFFFFFF).astype(np.uint32)
    
    # Extract bytes using vectorized operations
    bytes_out = np.empty(arr_float32.shape + (4,), dtype=np.uint8)
    bytes_out[..., 0] = (ints >> 24) & 0xFF
    bytes_out[..., 1] = (ints >> 16) & 0xFF
    bytes_out[..., 2] = (ints >> 8) & 0xFF
    bytes_out[..., 3] = ints & 0xFF
    
    return bytes_out

def uniform_uint8s_to_float32_array(bytes_in: np.ndarray) -> np.ndarray:
    """Converts a uint8 array of shape (*original_shape, 4) back to a float32 array with original shape."""
    assert bytes_in.dtype == np.uint8 and bytes_in.shape[-1] == 4
    
    # Combine bytes into uint32 integers
    ints = (bytes_in[..., 0].astype(np.uint32) << 24) | \
           (bytes_in[..., 1].astype(np.uint32) << 16) | \
           (bytes_in[..., 2].astype(np.uint32) << 8)  | \
            bytes_in[..., 3].astype(np.uint32)
    
    # Convert integers to float32 values between [0, 1]
    floats_out = ints.astype(np.float64) / 0xFFFFFFFF
    
    return floats_out.astype(np.float32)

def float16_array_to_uniform_uint8s(arr_float16: np.ndarray) -> np.ndarray:
    """Converts a float16 array in [0, 1] to uint8 array with shape (*original_shape, 2)."""
    assert arr_float16.dtype == np.float16

    # Scale to uint16 range
    ints = np.round(arr_float16 * 0xFFFF).astype(np.uint16)

    # Extract two bytes
    bytes_out = np.empty(arr_float16.shape + (2,), dtype=np.uint8)
    bytes_out[..., 0] = (ints >> 8) & 0xFF
    bytes_out[..., 1] = ints & 0xFF

    return bytes_out

def uniform_uint8s_to_float16_array(bytes_in: np.ndarray) -> np.ndarray:
    """Converts a uint8 array of shape (*original_shape, 2) to float16 array in [0, 1]."""
    assert bytes_in.dtype == np.uint8 and bytes_in.shape[-1] == 2

    # Reconstruct uint16 integers from bytes
    ints = (bytes_in[..., 0].astype(np.uint16) << 8) | bytes_in[..., 1].astype(np.uint16)

    # Normalize to [0, 1] and convert to float16
    floats_out = (ints.astype(np.float32) / 0xFFFF).astype(np.float16)

    return floats_out

def compress_with_jpeg(image, quality=40, mode = "dB"):
    if not features.check_codec("jpg_2000"):
        raise RuntimeError("JPEG 2000 codec not available in Pillow build.")
    
    # Convert to uint8 and RGB if needed
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)  # Convert grayscale to RGB

    pil_image = Image.fromarray(image).convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG2000", quality_mode=mode, quality_layers=[quality])
    return buffer.getvalue()

def decompress_jpeg(jpeg_bytes):
    buffer = io.BytesIO(jpeg_bytes)
    return np.array(Image.open(buffer))


def calculate_psnr(ground_truth: np.ndarray, reconstructed: np.ndarray) -> float:

    # Compute MSE on valid pixels only
    diff = ground_truth - reconstructed
    mse = np.mean(diff ** 2)

    if mse == 0:
        return float('inf')

    max_pixel_value = np.max(np.abs(ground_truth))
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

    return psnr

def calculate_psnr_masked(ground_truth: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:

    if mask.shape != ground_truth.shape[:2]:
        raise ValueError("Mask shape must match image spatial dimensions (H, W)")
    mask = mask.astype(bool)
    diff = (ground_truth - reconstructed)
    masked_diff = diff[mask]
    mse = np.mean(masked_diff ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = np.max(np.abs(ground_truth[mask]))
    psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

    return psnr

def calculate_psnr_masked_skimage(ground_truth: np.ndarray, reconstructed: np.ndarray, mask: np.ndarray) -> float:

    if mask.shape != ground_truth.shape[:2]:
        raise ValueError("Mask must have the same spatial dimensions as the image.")
    mask = mask.astype(bool)
    if ground_truth.ndim == 3:
        mask = np.stack([mask,mask,mask], axis = -1)

    gt_masked = ground_truth[mask]
    rec_masked = reconstructed[mask]
    data_range = np.max(gt_masked) - np.min(gt_masked)

    return skimage_psnr(gt_masked, rec_masked, data_range=data_range)

def depth_from_xyz(xyz_image, clip_near, clip_far, normalized = False):

    depth = xyz_image[:, :, 2]
    depth = np.clip(depth, clip_near, clip_far)
    if normalized:
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
    return depth.astype(np.float16)


def masked_blur(image, mask, kernel_size=(4, 4)):

    mask = mask.astype(np.uint8)
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask and image must have the same spatial dimensions")

    blurred = cv2.blur(image, kernel_size)

    if len(image.shape) == 3 and image.shape[2] == 3:
        mask_3ch = np.stack([mask]*3, axis=-1)
    else:
        mask_3ch = mask

    result = np.where(mask_3ch == 1, image, blurred)

    return result