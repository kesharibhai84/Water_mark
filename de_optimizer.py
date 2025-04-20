import numpy as np

def fitness(alpha_edge, alpha_smooth, LL, S_LL, watermark, edge_mask):
    # Sanity check to catch shape mismatches early
    assert S_LL.shape == watermark.shape == edge_mask.shape, (
        f"Shape mismatch:\nS_LL: {S_LL.shape}, "
        f"watermark: {watermark.shape}, edge_mask: {edge_mask.shape}"
    )

    adaptive_alpha = np.where(edge_mask, alpha_edge, alpha_smooth)
    S_LL_wm = S_LL + adaptive_alpha * watermark

    # PSNR calculation
    mse = np.mean((S_LL_wm - S_LL) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else 100

    # NC calculation (normalized correlation)
    nc_num = np.sum(watermark * (S_LL_wm - S_LL) / (adaptive_alpha + 1e-8))
    nc_den = np.sqrt(np.sum(watermark ** 2))
    nc = nc_num / (nc_den + 1e-8)

    return psnr + nc  # Can adjust weights if needed


def optimize_scaling_factors(LL, S_LL, watermark, edge_mask, pop_size=10, generations=20):
    # Ensure shape consistency before starting optimization
    assert S_LL.shape == watermark.shape == edge_mask.shape, (
        f"Shape mismatch:\nS_LL: {S_LL.shape}, "
        f"watermark: {watermark.shape}, edge_mask: {edge_mask.shape}"
    )

    population = np.random.uniform(0.01, 0.1, (pop_size, 2))  # [alpha_edge, alpha_smooth]

    for gen in range(generations):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + 0.8 * (b - c), 0.01, 0.1)
            cross_points = np.random.rand(2) < 0.7
            trial = np.where(cross_points, mutant, population[i])

            # Evaluate fitness
            fit_trial = fitness(trial[0], trial[1], LL, S_LL, watermark, edge_mask)
            fit_target = fitness(population[i][0], population[i][1], LL, S_LL, watermark, edge_mask)

            if fit_trial > fit_target:
                population[i] = trial

    best = max(population, key=lambda x: fitness(x[0], x[1], LL, S_LL, watermark, edge_mask))
    return best[0], best[1]
