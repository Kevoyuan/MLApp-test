def is_point_in_mask(point, mask):
    """
    Check if a point is in a mask.

    Parameters:
    point (tuple): The point to be checked.
    mask (numpy.array): The mask in which to check the point.

    Returns:
    bool: True if the point is in the mask, False otherwise.
    """
    x, y = point
    height = len(mask)
    width = len(mask[0]) if height > 0 else 0
    if 0 <= x < width and 0 <= y < height:
        return mask[y][x] == 1
    else:
        return False