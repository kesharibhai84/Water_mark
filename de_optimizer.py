import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.feature import local_binary_pattern

def calculate_content_mask(LL3):
    """
    Generate content-aware mask using edge detection and texture analysis
    """
    # Ensure LL3 is 2D
    if len(LL3.shape) > 2:
        LL3 = np.mean(LL3, axis=-1)
    
    # Convert to uint8 for LBP
    LL3_uint8 = ((LL3 - np.min(LL3)) / (np.max(LL3) - np.min(LL3)) * 255).astype(np.uint8)
    
    # Edge detection using Sobel
    edge_mask = sobel(LL3_uint8)
    edge_mask = (edge_mask - np.min(edge_mask)) / (np.max(edge_mask) - np.min(edge_mask))
    
    # Texture analysis using Local Binary Patterns
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(LL3_uint8, n_points, radius, method='uniform')
    texture_mask = ndimage.gaussian_filter(lbp, sigma=1.0)
    texture_mask = (texture_mask - np.min(texture_mask)) / (np.max(texture_mask) - np.min(texture_mask))
    
    # Combine edge and texture information
    content_mask = np.maximum(edge_mask, texture_mask)
    
    # Normalize to range [0.5, 1.5] to act as scaling multiplier
    content_mask = 0.5 + content_mask
    
    # Smooth the mask
    content_mask = ndimage.gaussian_filter(content_mask, sigma=1.0)
    
    return content_mask

def calculate_psnr(original, modified):
    """Calculate PSNR with proper scaling and error handling"""
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    
    # Normalize values to 0-1 range if needed
    if original.max() > 1:
        original = original / 255.0
    if modified.max() > 1:
        modified = modified / 255.0
        
    mse = np.mean((original - modified) ** 2)
    if mse < 1e-10:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_nc(watermark, extracted):
    """Calculate Normalized Correlation with proper normalization"""
    watermark = watermark.astype(np.float64)
    extracted = extracted.astype(np.float64)
    
    # Normalize to zero mean and unit variance
    watermark = (watermark - np.mean(watermark)) / (np.std(watermark) + 1e-10)
    extracted = (extracted - np.mean(extracted)) / (np.std(extracted) + 1e-10)
    
    return np.corrcoef(watermark.flatten(), extracted.flatten())[0,1]

def evaluate_watermark(params, LL3, S_LL, watermark, U_LL, V_LL):
    """Evaluate watermarking with content-aware embedding"""
    q, k = params
    
    # Generate content-aware mask
    content_mask = calculate_content_mask(LL3)
    
    # Calculate adaptive scaling factors
    singular_value_importance = np.exp(-k * np.arange(len(S_LL)) / len(S_LL))
    
    # Reshape for broadcasting
    scaling_factor = singular_value_importance.reshape(-1, 1)  # Shape: (64, 1)
    content_mask_reshaped = content_mask.reshape(1, -1)       # Shape: (1, 4096)
    
    # Create scaling matrix that matches S_LL shape
    scaling_matrix = q * np.outer(scaling_factor, content_mask_reshaped)
    scaling_matrix = scaling_matrix[:S_LL.shape[0], :S_LL.shape[1]]  # Ensure correct shape
    
    # Embed watermark
    S_LL_max = np.max(S_LL)
    S_LL_watermarked = S_LL + S_LL_max * scaling_matrix * watermark
    
    # Reconstruct
    LL3_watermarked = np.dot(U_LL, np.dot(S_LL_watermarked, V_LL.T))
    
    # Extract watermark
    denominator = S_LL_max * scaling_matrix
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    S_w_extracted = (S_LL_watermarked - S_LL) / denominator
    
    # Calculate metrics
    psnr_image = calculate_psnr(LL3, LL3_watermarked)
    psnr_watermark = calculate_psnr(watermark, S_w_extracted)
    nc = calculate_nc(watermark, S_w_extracted)
    
    return psnr_image, psnr_watermark, nc

def fitness_function(params, LL3, S_LL, watermark, U_LL, V_LL):
    """Adaptive fitness function for content-aware watermarking"""
    psnr_image, psnr_watermark, nc = evaluate_watermark(params, LL3, S_LL, watermark, U_LL, V_LL)
    
    # Target values for balanced performance
    target_psnr_image = 42.0
    target_psnr_watermark = 38.0
    
    # Calculate deviations from targets
    dev_image = abs(psnr_image - target_psnr_image)
    dev_watermark = abs(psnr_watermark - target_psnr_watermark)
    
    # Quality thresholds with content-aware considerations
    if psnr_image < 38 or psnr_watermark < 32 or nc < 0.9:
        return 0.0
    
    # Adaptive fitness calculation
    fitness = 10.0 / (1.0 + 0.6 * dev_image + 0.4 * dev_watermark)
    
    # Bonus for high correlation and balanced PSNRs
    if nc > 0.95 and max(dev_image, dev_watermark) < 5:
        fitness *= 1.2
    
    return fitness

def optimize_parameters(LL3, S_LL, watermark, U_LL, V_LL, pop_size=30, generations=40):
    """Content-aware optimization process"""
    dimension = 2
    population = np.random.rand(pop_size, dimension)
    
    # Adjusted parameter ranges for content-aware embedding
    # q: [0.008, 0.025], k: [0.6, 1.2]
    population[:, 0] = 0.008 + population[:, 0] * 0.017  # q
    population[:, 1] = 0.6 + population[:, 1] * 0.6      # k
    
    F = 0.6  # Mutation factor
    CR = 0.7 # Crossover rate
    
    best_solution = np.array([0.012, 0.9])  # Default solution
    best_fitness = -np.inf
    best_metrics = {'psnr_image': 0, 'psnr_watermark': 0, 'nc': 0}
    
    for gen in range(generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            mutant = np.clip(a + F * (b - c), [0.008, 0.6], [0.025, 1.2])
            cross_points = np.random.rand(dimension) < CR
            trial = np.where(cross_points, mutant, population[i])
            
            # Evaluate trial solution
            trial_fitness = fitness_function(trial, LL3, S_LL, watermark, U_LL, V_LL)
            target_fitness = fitness_function(population[i], LL3, S_LL, watermark, U_LL, V_LL)
            
            if trial_fitness > target_fitness:
                population[i] = trial
                if trial_fitness > best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()
                    
                    # Update metrics
                    psnr_image, psnr_watermark, nc = evaluate_watermark(best_solution, LL3, S_LL, watermark, U_LL, V_LL)
                    best_metrics['psnr_image'] = psnr_image
                    best_metrics['psnr_watermark'] = psnr_watermark
                    best_metrics['nc'] = nc
        
        if gen % 5 == 0:
            print(f"Generation {gen}: Best fitness = {best_fitness:.4f}")
            print(f"Current best metrics - Image PSNR: {best_metrics['psnr_image']:.2f} dB, "
                  f"Watermark PSNR: {best_metrics['psnr_watermark']:.2f} dB, "
                  f"NC: {best_metrics['nc']:.4f}")
        
        # Early stopping if good solution found
        if (38 <= best_metrics['psnr_image'] <= 45 and 
            35 <= best_metrics['psnr_watermark'] <= 42 and 
            best_metrics['nc'] >= 0.95):
            print(f"Found good solution at generation {gen}")
            break
    
    # If optimization failed to find a good solution, use a conservative default
    if best_fitness == -np.inf:
        best_solution = np.array([0.012, 0.9])
        
    return best_solution[0], best_solution[1]
