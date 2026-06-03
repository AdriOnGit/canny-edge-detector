"""
File containing utility functions for canny edge detector.
"""

import argparse
import matplotlib.pyplot as plt
import cv2


def parse_args():   
    """
    Parses the command line arguments.
    """     
    parser = argparse.ArgumentParser(
        description="Canny Edge Detector usage:",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i", "--image",    required = True,                help = "Path to input image")
    parser.add_argument("-s", "--size",     type = int,   default = 5,      help = "Gaussian kernel size")
    parser.add_argument("-g", "--sigma",    type = float, default = 1.0,    help = "Gaussian sigma")
    parser.add_argument("-l", "--low",      type = float, default = 0.1,    help = "Low threshold ratio")
    parser.add_argument("-u", "--high",     type = float, default = 0.3,    help = "High threshold ratio")

    return parser.parse_args()


def img_load_and_grayscale(img_path):
    """
    Takes an image path in input. The image format can be jpg or png.
    Loads the image with opencv and applies grayscale on the image.
    Returns original image and grayscaled image.
    """
    # Upload image in BGR.
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        raise FileNotFoundError(f"Image {img_path} does not exist.")
    
    # Apply grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_bgr, img_gray


def img_resize(img, max_width=800):
    """
    Resizes an image proportionally if its width exceeds max width.
    Returns resized image.
    """
    # Get height and width: shape[0] = height, shape[1] width, shape[2] color
    height, width = img.shape[:2]

    if width > max_width:
        ratio = max_width / float(width)
        new_dim = (max_width, int(height * ratio))
        # INTER_AREA downsamples the image by averaging neighboring pixels.
        img_resized = cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)
        
        return img_resized
    
    return img


def show_img (*images, titles=None):
    """
    Displays the passed images with opencv with the passed titles.
    """
    for i, img in enumerate(images):
        # Show title if given, if not then put image number.
        if titles and i < len(titles):
            title = titles[i] 
        else:
            title = f"Image {i+1}"
        
        cv2.imshow(title, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_results(images, titles, cols=3, filename=None):
    """
    Plots the selected images across a subplot grid with matplotlib.
    Allows option to save the image if a filename is given.
    """
    # Get the number of needed rows. Divide with // to round down the value.
    rows = (len(images) + cols - 1) // cols
    # Create the grid of subplots.
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    # Flatten the 2D axes array to 1D so we can iterate.
    #   [[ax1, ax2, ax3],   --->    [ax1, ax2, ax3, ax4, ax5, ax6]
    #   [ax4, ax5, ax6]]
    axes = axes.flatten()
    
    # Group by image and title and iterate showing image and setting title.
    for i, (img, title) in enumerate(zip(images, titles)):
        # Maps to gray when image is not RGB.
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        # Hides the x y axis for plots.
        axes[i].axis('off')
    
    # Hide any unused subplots.
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    # Automatically adjust spacing between subplots.
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    
    plt.show()
