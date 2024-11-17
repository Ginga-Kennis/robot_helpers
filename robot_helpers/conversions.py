import numpy as np

def map_cloud_to_grid(voxel_size, points, values):
    grid = np.zeros((40, 40, 40), dtype=np.float32)
    indices = np.round(points / voxel_size).astype(int)
    grid[tuple(indices.T)] = values.squeeze()
    return grid


def grid_to_map_cloud(voxel_size, grid, threshold=0.0):
    grid_shape = grid.shape
    grid_indices = np.indices(grid_shape).reshape(3, -1).T 
    grid_values = grid.flatten() 

    mask = grid_values >= threshold
    indices = grid_indices[mask]
    values = grid_values[mask]

    points = indices * voxel_size
    values = values[:, None] 

    return points, values
