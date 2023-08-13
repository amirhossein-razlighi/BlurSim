import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class BlurSim(nn.Module):
    """
    This is a custom loss function that calculates the loss between 2 blurred images.
    """

    def __init__(self, sobel_kernel_size=5, device="cpu", alpha=0.9):
        super(BlurSim, self).__init__()
        self.sobel_kernel_size = sobel_kernel_size
        self.device = device
        self.alpha = alpha

    def _sobel(self, img, kernel_size=5):
        if not isinstance(img, np.ndarray):
            img = np.array(img, dtype=np.float32)

        # Sobel Operator
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        # Calculate the magnitude of the gradient
        sobel = np.sqrt(sobelx**2 + sobely**2)

        # Calculate the direction of the gradient
        theta = np.arctan2(sobely, sobelx)

        # Standardize the magnitude of the gradient
        # sobel = (sobel / np.max(sobel)) * 255

        return torch.tensor(sobel).to(self.device)

    def _dft2D(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        try:
            rows, cols = img.shape
        except:
            raise Exception("Your Image should be grayscale!")

        m = cv2.getOptimalDFTSize(rows)
        n = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(
            img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=0
        )
        padded = np.float32(padded)

        dft = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum_3 = 1 * np.log(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        )

        return torch.tensor(dft_shift).to(self.device)

    def forward(self, img1, img2):
        """
        Calculates the loss between 2 blurred images.
        """
        sobel_diff = torch.abs(self._sobel(img1) - self._sobel(img2))
        sobel_diff /= sobel_diff.shape[0] * sobel_diff.shape[1]

        dft_diff = torch.abs(self._dft2D(img1) - self._dft2D(img2))
        dft_diff /= dft_diff.shape[0] * dft_diff.shape[1]

        sobel_diff = torch.norm(sobel_diff)
        dft_diff = torch.norm(dft_diff)

        return (1 - self.alpha) * dft_diff + self.alpha * sobel_diff
