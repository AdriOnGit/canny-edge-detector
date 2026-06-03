"""
File containing functions for each Canny edge detector step:
    1. Gaussian blur
    2. Sobel gradients
    3. Non-maximum suppression
    4. Double thresholding
    5. Hysteresis
"""

import numpy as np

def img_convolve(img, kernel):
    """
    Performs convolution step by sliding the given kernel over 
    the image's pixels and calculates the weighted sum.
    """
    # Get the image size and the kernel size. Grayscale images only have 2 channels.
    img_h, img_w = img.shape
    k_size = kernel.shape[0]
    # Calculate the amount of padding needed for the border pixels.
    pad = k_size // 2

    # Pad the image borders by reflecting.
    #   reflect: dcb|abcd|cba
    img_padded = np.pad(img, pad, mode="reflect")

    # Create image copy to put results in.
    # Keep float64 because we will insert weighted sums.
    img_output = np.zeros_like(img, dtype=np.float64)

    # Slide the kernel over every pixel.
    for i in range(img_h):
        for j in range(img_w):
            # Get the patch of pixels the kernel is on.
            kernel_patch = img_padded[i : i + k_size, j : j + k_size]
            # Perform weighted sum on the selected patch.
            img_output[i,j] = np.sum(kernel_patch * kernel) 

    return img_output


def gaussian_kernel(size=5, sigma=1.0):
    """
    Creates a 2D Gaussian kernel matrix for the Gaussian blur step.
    size: kernel size, use odd numbers (3, 5, 7...).
    sigma: standard deviation, controls the blur strength.
    """
    # Calculate the distance from the kernel center.
    kernel_center = size // 2
    # Create a 1D array of integer offsets from the center:
    #   e.g size=5 -> [-2, -1, 0, 1, 2]
    offset = np.arange(-kernel_center, kernel_center + 1)

    # Apply 1D Gaussian to the offset.
    gauss_1d = np.exp(-offset**2 / (2 * sigma**2))

    # Produce the 2D Gaussian kernel from two 1D Gaussians.
    # np.outer -> k[i][j] = gauss_1d[i] * gauss_1d[j]
    kernel = np.outer(gauss_1d, gauss_1d)

    # Return the normalized kernel.
    # array.sum adds all kernel values.
    # Divide the kernel by the value so that all kernel values sum to 1.
    return kernel / kernel.sum()


def gaussian_blur(img, size=5, sigma=1.0):
    """
    Performs the Gaussian blur step.
        1. Get the kernel from gaussian_kernel();
        2. Perform convolution with the gaussian kernel.
        3. Return blurred image
    """
    kernel = gaussian_kernel(size, sigma)
    img_blurred = img_convolve(img, kernel)

    # Clip the pixel values (0, 255) and return to uint8 for opencv.
    img_blurred = np.clip(img_blurred, 0, 255).astype(np.uint8)

    return img_blurred


def sobel_gradients(img):
    """
    Finds the edges in the image by calculating Sobel gradients.
        1. Creates the 2 Sobel kernels:
                 -1 0 1          -1 -2 -1
            Gx = -2 0 2 ,   Gy = 0  0  0
                 -1 0 1           1  2  1

        2. Slides the kernels over the blurred image.
        3. Returns calculated gradient magnitude (edges) and edge angle arrays.
    """
    # Create the Sobel kernels.
    # Define them as float64 for consistency with convolution step.
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    
    Ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float64)
    
    # Convolve the image with each kernel.
    Gx = img_convolve(img, Kx)
    Gy = img_convolve(img, Ky)

    # Compute the gradient magnitude with the Pythagorean theorem.
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Normalize the magnitude to get values in (0, 255) for images.
    # magnitude / magnitude.max() scales matrix to values in [0, 1].
    # Multiplying by 255 stretches it to [0, 255].
    img_edges = (magnitude / magnitude.max()) * 255

    # Compute the gradient direction in degrees.
    angles = np.arctan2(Gy, Gx) * (180 / np.pi)

    # Clip the angles to have them in range [0°, 180°].
    # Use numpy's boolean indexing:
    #   angle < 0 creates a boolean array.
    #   += 180 adds 180 where the condition is True.
    angles[angles < 0] += 180

    # Return as uint8 for image viewing.
    return img_edges.astype(np.uint8), angles


def nms(sobel_edges, angles):
    """
    Performs non-maximum suppression on each pixel to thin edges by
    keeping only the local maximum along the gradient direction.
        1. Bin each angle into one of the 4 possible directions (0°, 45°, 90°, 135°).
           To do this we split the directions into sections:
           0°        22.5°     67.5°     112.5°    157.5°    180°
            |---------|---------|---------|---------|---------|
                0°         45°       90°      135°       0°
        2. Keep the pixel only if it's the local maximum along the edge.
        3. Return the image with thinned edges.
    """
    img_h, img_w = sobel_edges.shape
    # Create image copy for output. Keep at float64 for calculations.
    img_thin_edges = np.zeros_like(sobel_edges, dtype=np.float64)

    for i in range(1, img_h -1):
        for j in range(1, img_w -1):

            angle = angles[i, j]

            # Bin the angle into the 4 possible directions.
            # 0° or 180° case - vertical edge |
            if (angle < 22.5) or (angle >= 157.5):
                # Left neighbor
                n1 = sobel_edges[i, j-1]
                # Right neighbor
                n2 = sobel_edges[i, j+1]
            # 45° case - diagonal edge /
            elif (angle >= 22.5) and (angle < 67.5):
                # Top right neighbor
                n1 = sobel_edges[i-1, j+1]
                # Top left neighbor
                n2 = sobel_edges[i+1, j-1]
            # 90° case - horizontal edge −
            elif (angle >= 67.5) and (angle < 112.5):
                # Top neighbor
                n1 = sobel_edges[i-1, j]
                # Bottom neighbor
                n2 = sobel_edges[i+1, j]
            # 135° case - diagonal edge \
            else:
                # Top left neighbor
                n1 = sobel_edges[i+1, j-1]
                # Bottom right neighbor
                n2 = sobel_edges[i-1, j+1]

            # Keep pixel only if it's the local maximum
            if sobel_edges[i, j] >= n1 and sobel_edges[i, j] >= n2:
                img_thin_edges[i, j] = sobel_edges[i, j]
            else:
                img_thin_edges[i, j] = 0
    
    return img_thin_edges.astype(np.uint8)
            

def double_threshold(thinned_edges, low_ratio=0.1, high_ratio=0.3):
    """
    Classifies each edge into strong, weak, or fake edges.
    The ratios must be tuned properly to get expected results: different ratios work
    for different image cases.
        1. Calculate the two threshold values.
        2. Define values for strong pixels and weak pixels for debugging visually.
        3. Apply the thresholds to the thinned edges.
        4. Return the thresholded image and the values for strong and weak pixels.
    """
    # Get thresholds by multiplying the maximum magnitude by the ratio.
    # This way, each threshold is adapted to each image.
    low_thr = thinned_edges.max() * low_ratio
    high_thr = thinned_edges.max() * high_ratio

    # Values for strong and weak pixels.
    strong_pix = 200
    weak_pix = 100

    # Image copy for output. Float64 for calculation purposes.
    img_dthr = np.zeros_like(thinned_edges, dtype=np.float64)

    # Apply thresholds to each edgge by using numpy's boolean indexing.
    # thinned_edges >= high creates a boolean array whose values = 1 when thinned_edges >= high.
    # The weak mask combines two boolean arrays via &. 
    # Values in both arrays must be = 1, else it is set as = 0.
    strong_mask = thinned_edges >= high_thr
    weak_mask = (thinned_edges >= low_thr) & (thinned_edges < high_thr)

    # Create the thresholded image by using the two masks.
    img_dthr[strong_mask] = strong_pix
    img_dthr[weak_mask] = weak_pix

    return img_dthr.astype(np.uint8), strong_pix, weak_pix


def hysteresis(img_dthr, strong_pix, weak_pix):
    """
    Promotes weak edges connected to strong edges and discards the isolated weak edges.
    We use a stack to start from strong pixels and propagate outward to find connected weak ones.
        1. Add all strong pixels to the stack.
        2. Pop from the stack and check its 8 neighbors.
        3. If neighbor is weak -> promote to strong and add to stack
        4. Repeat until stack is empty.
        5. Discard the isolated weak edges.
        6. Return the final edges.
    """
    img_h, img_w = img_dthr.shape
    # Copy the edge values to the output image.
    img_final_edges = img_dthr.copy()

    # Find positions of all strong pixels.
    strong_rows, strong_cols = np.where(img_final_edges == strong_pix)
    # Pair them into (row, col) tuples.
    strong_pos = zip(strong_rows, strong_cols)
    # Convert to list to append and pop.
    stack = list(strong_pos)

    while stack:
        # Current pixel position.
        i, j = stack.pop()
        
        # Check the 8 neighbors: di - row offset, dj - column offset.
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # Skip the pixel itself.
                if di == 0 and dj == 0: continue
                # Get the neighbor's position.
                ni = i + di
                nj = j + dj

                # Check the bounds before accessing.
                if 0 <= ni < img_h and 0 <= nj < img_w:
                    # Promote weak edge to strong
                    if img_final_edges[ni, nj] == weak_pix:
                        img_final_edges[ni, nj] = strong_pix
                        stack.append((ni, nj))
    
    # Discard remaining weak edges
    img_final_edges[img_final_edges == weak_pix] = 0

    return img_final_edges


def canny_edge_detection(grayscale_img, kernel_size=5, sigma=1.0, low_ratio=0.1, high_ratio=0.3):
    """
    Performs all Canny edge detection steps.
    Returns a dictionary containing the image after every step.
    """
    blurred = gaussian_blur(grayscale_img, kernel_size, sigma)
    sobel_edges, angles = sobel_gradients(blurred)
    thinned_edges = nms(sobel_edges, angles)
    dthr, strong_pix, weak_pix = double_threshold(thinned_edges, low_ratio, high_ratio)
    final_edges = hysteresis(dthr, strong_pix, weak_pix)
    
    # Store every image in a dictionary.
    images = {
        "blurred": blurred,
        "sobel": sobel_edges,
        "nms": thinned_edges,
        "dthr": dthr,
        "final": final_edges
    }

    return images
