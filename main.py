import cv2
from src.utilities import parse_args, img_load_and_grayscale, img_resize, plot_results
from src.canny import canny_edge_detection

def main():
    """
    Entrypoint for Canny Edge Detector.
    """
    args = parse_args()
    img_path = args.image
    
    original, grayscale = img_load_and_grayscale(img_path)
    original = img_resize(original)
    grayscale = img_resize(grayscale)
    
    # Perform Canny Edge Detection on the grayscaled image.
    images = canny_edge_detection(grayscale, args.size, args.sigma, args.low, args.high)
    # Perform opencv's Canny Edge Detection on the grayscaled image.
    opencv_canny = cv2.Canny(grayscale, 100, 200)

    # Convert cv2 BGR image to RGB for matplotlib.
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    plot_results(
        [original_rgb, images["blurred"], images["sobel"], images["nms"], images["dthr"], images["final"]],
        ["Original", "Blurred", "Sobel", "NMS", "Double Threshold", "Final Canny"],
        filename = "canny_comparison.png"
    )

    plot_results(
        [images["final"], opencv_canny],
        ["My Canny implementation", "OpenCV Canny implementation"],
        cols = 2,
        filename = "implementation_comparison.png"
    )


if __name__ =="__main__":
    main()