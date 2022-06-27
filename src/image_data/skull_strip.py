import cv2
import numpy as np


def rubbish_strip_skull(image):
    dir = "temp"
     # convert image to Grayscale and Black/White
    gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # invert gry
    gry = 255 - gry
    cv2.imwrite(dir+'/im_1_grayscale.png',gry)
    bw=cv2.threshold(gry, 127, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(dir+'/im_2_black_white.png',bw)

    # Use floodfill to identify outer shape of objects
    imFlood = bw.copy()
    h, w = bw.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(imFlood, mask, (0,0), 0)
    cv2.imwrite(dir+'/im_3_floodfill.png',imFlood)

    # Combine flood filled image with original objects
    imFlood[np.where(bw==0)]=255
    cv2.imwrite(dir+'/im_4_mixed_floodfill.png',imFlood)

    # Invert output colors
    imFlood=~imFlood
    cv2.imwrite(dir+'/im_5_inverted_floodfill.png',imFlood)

    # Find objects and draw bounding box
    cnts, _ = cv2.findContours(imFlood, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        bob = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save final image
    cv2.imwrite(dir+'/im_6_output.png',image)


def strip_skull(image):
    """ Strip skull 
    
    Args:
        image (np.ndarray): image to strip skull from
    """
    dir = "temp"
    print("[INFO] STRIPPING SKULL")
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert gray
    # Apply median filtering with a window of size 3×3 to the input image.
    median = cv2.medianBlur(gray, 3)
    # Compute the initial mean intensity value Ti of the image.
    Ti = np.mean(median)
    # Identify the top, bottom, left, and right pixel locations, from where brain skull starts in the image, considering gray values of the skull are greater than Ti.
    top = np.where(median > Ti)[0][0]
    bottom = np.where(median > Ti)[0][-1]
    left = np.where(median > Ti)[1][0]
    right = np.where(median > Ti)[1][-1]
    # Plot top, bottom, left, and right pixel locations on the image.
    cv2.circle(image, (left, top), 1, (0, 0, 255), 2)
    cv2.circle(image, (right, top), 1, (0, 0, 255), 2)
    cv2.circle(image, (left, bottom), 1, (0, 0, 255), 2)
    cv2.circle(image, (right, bottom), 1, (0, 0, 255), 2)
    # Save image
    cv2.imwrite(dir+'/plotted.png',image)

    # skull = image[top:bottom, left:right]
   
    # Form a rectangle using the top, bottom, left, and right pixel locations.

    # Compute the final mean value Tf of the brain using the pixels located within the rectangle.

    # Approximate the region of brain membrane or meninges that envelop the brain, based on the assumption that the intensity of skull is more than Tf and that of membrane is less than Tf

    # Set the average intensity value of membrane as the threshold value T.

    # Convert the given input image into binary image using the threshold T.

    # Apply a 13×13 opening morphological operation to the binary image in order to separate the skull from the brain completely.

    # Find the largest connected component and consider it as brain.

    # Finally, apply a 21×21 closing morphological operation to fill the gaps within and along the periphery of the intracranial region.
    pass 


def main():
    """Main"""
    # test strip_skull
    image = cv2.imread("plots/F/AD/S61715.jpg")
    print(f"[INFO] Image Loaded: {image.shape}")
    strip_skull(image)


if __name__ == "__main__":
    main()