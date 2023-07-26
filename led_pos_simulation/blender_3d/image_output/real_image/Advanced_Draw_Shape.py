import cv2
import numpy as np

drawing = False
mode = True # if True, draw rectangle. else, draw circle
ix,iy = -1,-1
rx,ry,rw,rh = -1,-1,-1,-1

# Mouse callback function
def draw_shape(event,x,y,flags,param):
    global ix,iy,drawing,mode,rx,ry,rw,rh

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
                rx,ry,rw,rh = ix,iy,abs(x-ix),abs(y-iy)
            else:
                cv2.circle(img,(ix,iy),int(np.sqrt((x-ix)**2 + (y-iy)**2)),(0,255,0),1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
        else:
            cv2.circle(img,(ix,iy),int(np.sqrt((x-ix)**2 + (y-iy)**2)),(0,255,0),1)

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_shape)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27: # ESC key to quit
        break

cv2.destroyAllWindows()

# Now, let's check if a point is inside the drawn rectangle
point = (250, 250) # Change this to the point you want to check

if (rx < point[0] < rx+rw) and (ry < point[1] < ry+rh):
    print("The point is inside the rectangle!")
else:
    print("The point is not inside the rectangle!")
