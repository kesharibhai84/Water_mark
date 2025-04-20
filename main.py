import cv2
import numpy as np
from watermark_utils import (
    rgb2yiq, yiq2rgb, get_edge_mask, dwt2, idwt2, svd2, isvd2,
    arnold_scramble, arnold_unscramble, resize_image
)
from de_optimizer import optimize_scaling_factors

# Load images
host_img = cv2.imread('images/host.png')
watermark_img = cv2.imread('images/watermark.png', 0)

# Convert host to YIQ and get Y channel
yiq_img = rgb2yiq(host_img.astype(np.float32))
Y_channel = yiq_img[:, :, 0]

# DWT on Y channel
LL, (LH, HL, HH) = dwt2(Y_channel)

# Edge mask for adaptive embedding (resize to match DWT output)
edge_mask_full = get_edge_mask(Y_channel)
edge_mask = cv2.resize(edge_mask_full, (LL.shape[1], LL.shape[0]), interpolation=cv2.INTER_NEAREST)

# SVD on LL
U_LL, S_LL, V_LL = svd2(LL)
assert S_LL.shape == LL.shape, f"Mismatch: {S_LL.shape} vs {LL.shape}"

# Scramble and resize watermark to match S_LL shape
wm_resized = resize_image(watermark_img, S_LL.shape)
watermark_scrambled = arnold_scramble(wm_resized, times=10)
assert watermark_scrambled.shape == S_LL.shape, f"Watermark shape mismatch: {watermark_scrambled.shape} vs {S_LL.shape}"

# Differential Evolution to get optimal scaling factors
alpha_edge, alpha_smooth = optimize_scaling_factors(LL, S_LL, watermark_scrambled, edge_mask)

# Create adaptive alpha mask
adaptive_alpha = np.where(edge_mask[:S_LL.shape[0], :S_LL.shape[1]], alpha_edge, alpha_smooth)

# Embed watermark
S_LL_watermarked = S_LL + adaptive_alpha * watermark_scrambled
LL_watermarked = isvd2(U_LL, S_LL_watermarked, V_LL)

# Inverse DWT and reconstruct image
watermarked_Y = idwt2(LL_watermarked, (LH, HL, HH))
yiq_img[:, :, 0] = watermarked_Y
watermarked_rgb = yiq2rgb(yiq_img)

# Convert to 8-bit before saving
watermarked_rgb = np.clip(watermarked_rgb, 0, 255).astype(np.uint8)

# Save the watermarked image
cv2.imwrite('images/watermarked.png', watermarked_rgb)

# Extraction
LL_wm, (_, _, _) = dwt2(watermarked_Y)
U_LL_wm, S_LL_wm, V_LL_wm = svd2(LL_wm)
extracted_watermark = (S_LL_wm - S_LL) / adaptive_alpha
extracted_watermark = arnold_unscramble(extracted_watermark, times=10)

# Convert extracted watermark to 8-bit and save
extracted_watermark = np.clip(extracted_watermark, 0, 255).astype(np.uint8)
cv2.imwrite('images/extracted_watermark.png', extracted_watermark)
