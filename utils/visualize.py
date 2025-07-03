import numpy as np
import matplotlib.pyplot as plt



def visualize_fusion(img1, img2, combined, rec1, rec2, titles=None):
    if img1.dtype != np.float32 and img1.dtype != np.float16:
        img1 = img1.astype(np.float32) / 255
    if img2.dtype != np.float32 and img2.dtype != np.float16:
        img2 = img2.astype(np.float32) / 255
    if rec1.dtype != np.float32 and rec1.dtype != np.float16:
        rec1 = rec1.astype(np.float32) / 255
    if rec2.dtype != np.float32 and rec2.dtype != np.float16:
        rec2 = rec2.astype(np.float32) / 255
    if combined.dtype != np.float32 and combined.dtype != np.float16:
        combined_disp = combined.astype(np.float32) / 255
    else:
        combined_disp = combined

    if titles is None:
        titles = [
            "Original Image 1",
            "Original Image 2",
            "Combined Packed Image",
            "Reconstructed Image 1",
            "Reconstructed Image 2"
        ]

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(img1, cmap='gray')
    axs[0].set_title(titles[0])
    axs[1].imshow(img2, cmap='gray')
    axs[1].set_title(titles[1])
    axs[2].imshow(combined_disp, cmap='gray')
    axs[2].set_title(titles[2])
    axs[3].imshow(rec1, cmap='gray')
    axs[3].set_title(titles[3])
    axs[4].imshow(rec2, cmap='gray')
    axs[4].set_title(titles[4])

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def fft_image(ch):
    # Forward 2D FFT
    fft2 = np.fft.fft2(ch)
    fft2_shifted = np.fft.fftshift(fft2)

    plt.figure(figsize=(6, 6))
    plt.title("Original Channel")
    plt.imshow(ch, cmap='gray')
    plt.axis('off')
    plt.show()

    # Show magnitude spectrum (for visualization only)
    plt.figure(figsize=(6, 6))
    plt.title("FFT Magnitude Spectrum")
    plt.imshow(np.log(np.abs(fft2_shifted) + 1), cmap='gray')
    plt.axis('off')
    plt.show()

    # Inverse FFT (Perfect reconstruction)
    ifft2 = np.fft.ifft2(fft2)
    reconstructed = np.real(ifft2)

    # Display reconstructed result
    plt.figure(figsize=(6, 6))
    plt.title("Perfectly Reconstructed Channel")
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_error(ground_truth, reconstructed):
    error = reconstructed - ground_truth

    plt.figure(figsize=(12, 5))

    plt.title("Error: Recovered - Ground Truth")
    plt.imshow(np.arcsinh(error), cmap='seismic')
    plt.colorbar()
    plt.axis('off')

    plt.show()