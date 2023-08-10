from Advanced_Function import *

# Load the image
# src = cv2.imread(f"{script_dir}/card_seg.png")
src = cv2.imread(f"{script_dir}/blob_seg.png")
if src is None:
    print("Could not open or find the image!")
    exit()

# Show the source image
cv2.imshow("Source Image", src)

# Change the background from white to black
mask = cv2.inRange(src, (255, 255, 255), (255, 255, 255))
src[mask == 255] = [0, 0, 0]
cv2.imshow("Black Background Image", src)

# Create a kernel for sharpening
kernel = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
], dtype=np.float32)

# Apply laplacian filtering
imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian

# Convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
cv2.imshow("New Sharped Image", imgResult)

# Create binary image from source image
bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Binary Image", bw)

# Perform the distance transform algorithm
dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow("Distance Transform Image", dist)

# Threshold to obtain the peaks
_, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
kernel1 = np.ones((3, 3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)
cv2.imshow("Peaks", dist)

# Create the CV_8U version of the distance image
dist_8u = dist.astype('uint8')

# Find total markers
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
markers = np.zeros(dist.shape, dtype=np.int32)

for i, contour in enumerate(contours):
    cv2.drawContours(markers, [contour], 0, i+1, -1)

# Draw the background marker
cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
markers_8u = (markers * 10).astype('uint8')
cv2.imshow("Markers", markers_8u)

# Perform the watershed algorithm
cv2.watershed(imgResult, markers)
mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)

# Generate random colors
colors = []
for i in range(len(contours)):
    colors.append(list(np.random.randint(0, 256, size=3)))

# Create the result image
dst = np.zeros_like(src)

for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i, j]
        if 0 < index <= len(contours):
            dst[i, j, :] = colors[index-1]

cv2.imshow("Final Result", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
