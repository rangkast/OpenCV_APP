from Advanced_Function import *


def make_video(camera_devices):
    # Select the first camera device
    camera_port = camera_devices[0]['port']

    # Open the video capture
    cap = cv2.VideoCapture(camera_port)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    out = cv2.VideoWriter('output.mkv', fourcc, 60.0, (1280, 960))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' instead of 'X264'
    # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 960))
    recording = False
    if cap.isOpened():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            draw_frame = frame.copy()
            if not ret:
                print("Unable to capture video")
                break

            # Convert the frame to grayscale
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            center_x, center_y = 1280 // 2, 960 // 2
            cv2.line(draw_frame, (0, center_y), (1280, center_y), (255, 255, 255), 1)
            cv2.line(draw_frame, (center_x, 0), (center_x, 960), (255, 255, 255), 1)     

            # Start/Stop recording
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                if not recording:
                    print('Start Recording')                
                    recording = True
            elif key & 0xFF == ord('q'):
                if recording:
                    print('Stop Recording')
                    out.release()
                    recording = False
            elif key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break
            elif key & 0xFF == ord('c'): 
                # Capture current frame
                cv2.imwrite('start_capture.jpg', frame)
                print('Image saved as capture.jpg')
            # Write the frame to file if recording
            if recording and out is not None:
                print('writing...')
                out.write(frame)
            

            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)

    if out is not None:
        out.release()
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())


    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    
    make_video(camera_devices)
