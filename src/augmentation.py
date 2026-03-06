import numpy as np

def jitter(x, sigma=0.03):
    # Add random noise
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # Scale by a random factor
    factor = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], x.shape[2]))
    return np.multiply(x, factor[:, np.newaxis, :])

def permutation(x, max_segments=5, seg_mode="equal"):
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments + 1)
    
    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        
        # Fix: Shuffle indices instead of the list of arrays directly to handle uneven splits
        perm_indices = np.random.permutation(len(splits))
        shuffled_splits = [splits[i] for i in perm_indices]
        warp = np.concatenate(shuffled_splits).ravel()
        return x[:, warp, :]
    else:
        return x

def time_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    # x is (N, T, F)
    orig_steps = np.arange(x.shape[1])
    
    # 6 knot points distributed evenly across time
    knot_points = np.linspace(0, x.shape[1]-1, knot+2)
    
    # Random magnitudes at knots
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[2]):
            # Interpolate random magnitudes across time
            warper = CubicSpline(knot_points, random_warps[i, :, j])(orig_steps)
            ret[i, :, j] = x[i, :, j] * warper
            
    return ret

def augment_data(X, y):
    # Apply augmentations
    # 1. Jitter
    X_jitter = jitter(X)
    # 2. Scaling
    X_scale = scaling(X)
    # 3. Simple Magnitude Warp/Time Warp (Simulated via scaling varying over time)
    X_warp = time_warp(X)
    
    # Combine original and augmented data (4x size now)
    X_aug = np.concatenate((X, X_jitter, X_scale, X_warp))
    y_aug = np.concatenate((y, y, y, y))
    
    return X_aug, y_aug
