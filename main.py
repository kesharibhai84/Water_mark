import cv2
import numpy as np
from watermark_utils import (
    rgb2yiq, yiq2rgb, dwt2, idwt2, svd2, isvd2,
    arnold_scramble, arnold_unscramble, resize_image
)
from de_optimizer import optimize_parameters

def apply_3level_dwt(Y):
    # Apply 3-level DWT
    LL1, (LH1, HL1, HH1) = dwt2(Y)
    LL2, (LH2, HL2, HH2) = dwt2(LL1)
    LL3, (LH3, HL3, HH3) = dwt2(LL2)
    return LL3, (LH3, HL3, HH3), (LH2, HL2, HH2), (LH1, HL1, HH1)

def inverse_3level_dwt(LL3, level3_coeffs, level2_coeffs, level1_coeffs):
    # Apply 3-level inverse DWT
    LL2 = idwt2(LL3, level3_coeffs)
    LL1 = idwt2(LL2, level2_coeffs)
    Y = idwt2(LL1, level1_coeffs)
    return Y

def normalize_array(arr):
    """Normalize array to [0,1] range with safety checks"""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val > 1e-10:
        return (arr - min_val) / (max_val - min_val)
    return np.zeros_like(arr)

# Load images
host_img = cv2.imread('images/host.png')
watermark_img = cv2.imread('images/watermark.png', 0)

print("Original watermark shape:", watermark_img.shape)
print("Host image shape:", host_img.shape)

# Convert host to YIQ and get Y channel
yiq_img = rgb2yiq(host_img.astype(np.float32))
Y_channel = yiq_img[:, :, 0]

# Apply 3-level DWT on Y channel
LL3, level3_coeffs, level2_coeffs, level1_coeffs = apply_3level_dwt(Y_channel)
print("LL3 shape after 3-level DWT:", LL3.shape)

# SVD on LL3
U_LL, S_LL, V_LL = svd2(LL3)
print("S_LL shape after SVD:", S_LL.shape)

# Store original S_LL for extraction
S_LL_original = S_LL.copy()

# Process watermark
# First resize watermark to match LL3 size
wm_resized = cv2.resize(watermark_img, (LL3.shape[1], LL3.shape[0]))
watermark_normalized = normalize_array(wm_resized.astype(float))

# Store original watermark for comparison
original_watermark = watermark_normalized.copy()

# Apply Arnold scrambling
watermark_scrambled = arnold_scramble(watermark_normalized, times=10)
print("Watermark shape after scrambling:", watermark_scrambled.shape)

# Apply SVD on scrambled watermark
U_w, S_w, V_w = svd2(watermark_scrambled)
print("S_w shape after SVD:", S_w.shape)

# Store SVD components of watermark
U_w_original = U_w.copy()
V_w_original = V_w.copy()

# Normalize singular values
S_LL_max = np.max(S_LL)
S_w_normalized = normalize_array(S_w)

# Optimizing embedding parameters using DE
print("Optimizing embedding parameters using DE...")
q, k = optimize_parameters(LL3, S_LL, S_w_normalized, U_LL, V_LL)
print(f"Optimized parameters: q = {q:.4f}, k = {k:.4f}")

# Calculate adaptive scaling factor using optimized parameters
singular_value_importance = np.exp(-k * np.arange(len(S_LL)) / len(S_LL))
scaling_factor = q * singular_value_importance

# Embed watermark
S_LL_watermarked = S_LL + np.max(S_LL) * scaling_factor[:, np.newaxis] * S_w_normalized

print("Embedding Info:")
print(f"Max scaling factor: {np.max(scaling_factor):.6f}")
print(f"Min scaling factor: {np.min(scaling_factor):.6f}")
print(f"Max singular value change: {np.max(np.abs(S_LL_watermarked - S_LL)):.2f}")

# Reconstruct watermarked image
LL3_watermarked = isvd2(U_LL, S_LL_watermarked, V_LL)

# Apply inverse 3-level DWT
watermarked_Y = inverse_3level_dwt(LL3_watermarked, level3_coeffs, level2_coeffs, level1_coeffs)

# Reconstruct color image
yiq_img[:, :, 0] = watermarked_Y
watermarked_rgb = yiq2rgb(yiq_img)

# Save watermarked image
watermarked_rgb = np.clip(watermarked_rgb, 0, 255).astype(np.uint8)
cv2.imwrite('images/watermarked.png', watermarked_rgb)

# Calculate PSNR
mse = np.mean((host_img - watermarked_rgb) ** 2)
psnr = 20 * np.log10(255 / np.sqrt(mse))
print("\nPSNR of watermarked image:", psnr)

# Extraction process
print("\nExtraction process:")
# Convert watermarked image to YIQ
watermarked_yiq = rgb2yiq(watermarked_rgb.astype(np.float32))
watermarked_Y = watermarked_yiq[:, :, 0]

# Apply 3-level DWT
LL3_wm, _, _, _ = apply_3level_dwt(watermarked_Y)

# Apply SVD
U_wm, S_wm, V_wm = svd2(LL3_wm)

# Extract watermark
S_w_extracted = (S_wm - S_LL_original) / (S_LL_max * scaling_factor[:, np.newaxis] + 1e-10)

# Reconstruct watermark using original U and V matrices
extracted_watermark = isvd2(U_w_original, S_w_extracted, V_w_original)

# Unscramble
extracted_watermark = arnold_unscramble(extracted_watermark, times=10)

# Normalize extracted watermark
extracted_watermark = normalize_array(extracted_watermark)

# Calculate similarity
correlation = np.corrcoef(original_watermark.flatten(), extracted_watermark.flatten())[0,1]
print("Correlation between original and extracted watermark:", correlation)

# Convert to 8-bit and save
extracted_watermark = (extracted_watermark * 255).astype(np.uint8)
cv2.imwrite('images/extracted_watermark.png', extracted_watermark)

# Calculate PSNR for watermark
mse_watermark = np.mean((original_watermark - extracted_watermark.astype(float)/255) ** 2)
psnr_watermark = 20 * np.log10(1 / np.sqrt(mse_watermark))

# Print quality metrics
print("\nQuality Metrics:")
print(f"Watermarked Image PSNR: {psnr:.2f} dB")
print(f"Extracted Watermark PSNR: {psnr_watermark:.2f} dB")
print(f"Watermark Correlation: {correlation:.4f}")

# Save comparison visualization
comparison = np.hstack([
    cv2.resize(watermark_img, (256, 256)),
    cv2.resize(extracted_watermark, (256, 256))
])
cv2.imwrite('images/watermark_comparison.png', comparison)
