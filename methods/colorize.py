import huecodec as hc
import numpy as np
from utils.utils import float16_array_to_uniform_uint8s, uniform_uint8s_to_float16_array
import cv2


def intel_hue_encode(depth, d_min, d_max, inv_depth = False):
    depth_colorized = hc.depth2rgb(depth, zrange=(d_min, d_max))
    return depth_colorized

def intel_hue_decode(depth_hue, d_min, d_max, inv_depth = False):
    depth_rec = hc.rgb2depth(depth_hue, zrange=(d_min, d_max))
    depth_rec = np.nan_to_num(depth_rec, nan=0.0)
    depth_rec = np.where(
        depth_rec < (d_min+0.00001), 0.0,
        np.where(depth_rec > (d_max-0.00001), 1.0, depth_rec)
    )
    return depth_rec

def bytesplit_encode(image_2d_float16: np.ndarray, num_ch = 2) -> np.ndarray:

    assert image_2d_float16.dtype == np.float16
    assert image_2d_float16.ndim == 2

    bytes_2ch = float16_array_to_uniform_uint8s(image_2d_float16)
    r = bytes_2ch[..., 0]
    g = bytes_2ch[..., 1]
    zeros = np.zeros_like(g, dtype = np.uint8)
    if num_ch == 2:
        rgb_image = np.stack([r, g, g], axis=-1).astype(np.uint8)
    elif num_ch == 1:
        rgb_image = np.stack([r, g, zeros], axis=-1).astype(np.uint8)
    return rgb_image

def bytesplit_decode(rgb_image: np.ndarray) -> np.ndarray:

    assert rgb_image.dtype == np.uint8
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3

    msb = rgb_image[..., 0]
    lsb = rgb_image[..., 1]

    bytes_2ch = np.stack([msb, lsb], axis=-1)
    float16_img = uniform_uint8s_to_float16_array(bytes_2ch)
    return float16_img

def quantize_encode(image_float16: np.ndarray, num_ch = 3) -> np.ndarray:

    assert image_float16.dtype == np.float16
    assert image_float16.ndim == 2

    image_uint8 = np.clip(image_float16 * 255.0, 0, 255).astype(np.uint8)

    zeros = np.zeros_like(image_uint8)
    if num_ch ==1:
        rgb_image = np.stack([image_uint8, zeros, zeros], axis=-1)
    elif num_ch ==2:
        rgb_image = np.stack([image_uint8, image_uint8, zeros], axis=-1)
    elif num_ch ==3:
        rgb_image = np.stack([image_uint8, image_uint8, image_uint8], axis=-1)
    return rgb_image

def quantize_decode(rgb_image: np.ndarray) -> np.ndarray:

    assert rgb_image.dtype == np.uint8
    assert rgb_image.ndim == 3 and rgb_image.shape[2] == 3

    grayscale = rgb_image[..., 0].astype(np.float16) / 255.0
    return grayscale.astype(np.float16)





#################
# Adapting Standard Video Codecs for Depth Streaming
# Fabrizio Pece; Jan Kautz; Tim Weyrich
#################


def _ha(l, p):

    frac = np.mod(l / (p/2.0), 2.0)
    return np.where(frac <= 1.0, frac, 2.0 - frac)

def _hb(l, p):

    frac = np.mod((l - p/4.0) / (p/2.0), 2.0)
    return np.where(frac <= 1.0, frac, 2.0 - frac)

def psk_encode(depth_2d: np.ndarray) -> np.ndarray:

    w_ = 2.0**13
    p = 512.0 / w_

    depth_clamped = np.clip(depth_2d, 0, w_ - 1).astype(np.float32)
    l = (depth_clamped + 0.5) / w_
    Ha = _ha(l, p)
    Hb = _hb(l, p)

    l_uint8  = (l  * 255.0).astype(np.uint8)
    ha_uint8 = (Ha * 255.0).astype(np.uint8)
    hb_uint8 = (Hb * 255.0).astype(np.uint8)

    rgb = np.stack([l_uint8, ha_uint8, hb_uint8], axis=-1)

    return rgb

def _m(l, p=(512.0/(2.0**16))):

    val = ((l/p)- 0.5) * 4.0
    return (np.floor(val) % 4).astype(np.int32)

def _L0(l, p):

    x = l - (p / 8.0)
    modp = np.mod(x, p)
    mm = _m(l)
    return l - modp + ((p / 4.0)*mm) - (p / 8.0)

def _delta(l, ha, hb, p):

    mm = _m(l)
    half_p = p / 2.0
    out = np.zeros_like(l, dtype=np.float32)

    cond0 = (mm == 0)
    cond1 = (mm == 1)
    cond2 = (mm == 2)
    cond3 = (mm == 3)

    out[cond0] = half_p * ha[cond0]
    out[cond1] = half_p * hb[cond1]
    out[cond2] = half_p * (1.0 - ha[cond2])
    out[cond3] = half_p * (1.0 - hb[cond3])
    return out

def psk_decode(rgb: np.ndarray) -> np.ndarray:

    w_ = 2.0**16
    p = 512.0 / w_

    l  = rgb[..., 0].astype(np.float32)/255.0
    ha = rgb[..., 1].astype(np.float32)/255.0
    hb = rgb[..., 2].astype(np.float32)/255.0
    l0 = _L0(l, p)

    d_small = l0 + _delta(l, ha, hb, p)
    depth_reconstructed = (w_ * d_small)

    depth_reconstructed = np.clip(depth_reconstructed, 0, w_-1)

    return depth_reconstructed

#################
# Range demultiplexing
# Fabrizio Nenci; Luciano Spinello; Cyrill Stachniss
#################



def range_demultiplexing_encode(
    depth_map: np.ndarray,
    intervals: list
) -> np.ndarray:
    
    if len(intervals) != 3:
        raise ValueError("Exactly 3 intervals must be provided.")
    
    (low0, high0), (low1, high1), (low2, high2) = intervals
    H, W = depth_map.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(depth_map) & (depth_map >= 0)

    mask0 = valid_mask & (depth_map >= low0) & (depth_map < high0)
    fraction0 = (depth_map[mask0] - low0) / (high0 - low0)
    fraction0 = np.clip(fraction0, 0, 1)
    rgb_image[mask0, 0] = np.round(255 * fraction0).astype(np.uint8)

    mask1 = valid_mask & (depth_map >= low1) & (depth_map < high1)
    fraction1 = (depth_map[mask1] - low1) / (high1 - low1)
    fraction1 = np.clip(fraction1, 0, 1)
    rgb_image[mask1, 1] = np.round(255 * fraction1).astype(np.uint8)

    mask2 = valid_mask & (depth_map >= low2) & (depth_map <= high2)
    fraction2 = (depth_map[mask2] - low2) / (high2 - low2)
    fraction2 = np.clip(fraction2, 0, 1)
    rgb_image[mask2, 2] = np.round(255 * fraction2).astype(np.uint8)

    return rgb_image


def range_demultiplexing_decode(
    rgb_image: np.ndarray,
    intervals: list
) -> np.ndarray:
    
    if len(intervals) != 3:
        raise ValueError("Exactly 3 intervals must be provided.")

    (low0, high0), (low1, high1), (low2, high2) = intervals
    H, W, _ = rgb_image.shape
    depth_map = np.zeros((H, W), dtype=np.float32)
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    maskR = (R > 0)
    fractionR = R[maskR].astype(np.float32) / 255.0
    depth_map[maskR] = fractionR * (high0 - low0) + low0

    maskG = (G > 0)
    fractionG = G[maskG].astype(np.float32) / 255.0
    depth_map[maskG] = fractionG * (high1 - low1) + low1

    maskB = (B > 0)
    fractionB = B[maskB].astype(np.float32) / 255.0
    depth_map[maskB] = fractionB * (high2 - low2) + low2

    return depth_map


def encode_depth_map(depth_map: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Encode a depth map using a specified method.

    Parameters:
        depth_map (np.ndarray): Input 2D depth map (float16 or float32 depending on method).
        method (str): Encoding method name.
        **kwargs: Additional method-specific parameters.

    Returns:
        np.ndarray: Encoded RGB image or byte array.
    """
    if method == "intel_hue":
        return intel_hue_encode(depth_map, **kwargs)
    elif method == "bytesplit":
        return bytesplit_encode(depth_map, **kwargs)
    elif method == "quantize":
        return quantize_encode(depth_map, **kwargs)
    elif method == "psk":
        return psk_encode(depth_map, **kwargs)
    elif method == "range_demux":
        return range_demultiplexing_encode(depth_map, **kwargs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
    

def decode_depth_map(encoded_data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Decode encoded depth representation back to a depth map.

    Parameters:
        encoded_data (np.ndarray): Encoded RGB image or byte array.
        method (str): Decoding method name.
        **kwargs: Additional method-specific parameters.

    Returns:
        np.ndarray: Decoded depth map.
    """
    if method == "intel_hue":
        return intel_hue_decode(encoded_data, **kwargs)
    elif method == "bytesplit":
        return bytesplit_decode(encoded_data)
    elif method == "quantize":
        return quantize_decode(encoded_data)
    elif method == "psk":
        return psk_decode(encoded_data)
    elif method == "range_demux":
        return range_demultiplexing_decode(encoded_data, **kwargs)
    else:
        raise ValueError(f"Unknown decoding method: {method}")