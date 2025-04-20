import cv2
import numpy as np

# Load the original and extracted watermark images
def load_images():
    original = cv2.imread('images/watermark.png', 0)
    extracted = cv2.imread('images/extracted_watermark.png', 0)
    return original, extracted

# Calculate Normalized Correlation
def calculate_nc(original, extracted):
    original = original.astype(np.float32)
    extracted = extracted.astype(np.float32)
    nc = np.sum(original * extracted) / np.sqrt(np.sum(original**2) * np.sum(extracted**2))
    return nc

# Display images and NC value
def display_results(original, extracted, nc):
    cv2.imshow('Original Watermark', original)
    cv2.imshow('Extracted Watermark', extracted)
    print(f'Normalized Correlation (NC): {nc:.4f}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    original, extracted = load_images()
    nc = calculate_nc(original, extracted)
    display_results(original, extracted, nc) 