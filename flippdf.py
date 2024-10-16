# NOT ACTUAL PROJECT, USED FOR REFERENCE

# Source: https://www.geeksforgeeks.org/how-to-extract-images-from-pdf-in-python/
import cv2
import fitz
import numpy as np

# Open the PDF document
doc = fitz.open("THE-VERY-HUNGRY-CATERPILLAR.pdf")
# Initialize the current page index
current_page = 0

while True:
    # Extract the current page as an image
    page = doc.load_page(current_page)
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:  # Handle images with an alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Display the image
    cv2.imshow("Page", img)

    # Handle user input for navigation
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n') and current_page < doc.page_count - 1:
        current_page += 1
    elif key == ord('p') and current_page > 0:
        current_page -= 1

cv2.destroyAllWindows()