'''

start here: https://learnopencv.com/moving-object-detection-with-opencv/

then need to improve this in case of walking camera. Right now, it will break because too much object detect because of camera movement.


'''
import cv2 as cv

video_path = "traffic.mp4"

cap = cv.VideoCapture(video_path)

backSub = cv.createBackgroundSubtractorMOG2()

if not cap.isOpened():
    print("Video not loaded")
    exit()
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("video capture ended")
        break

    # apply background removing
    fgMask = backSub.apply(frame)
    
    # Find contours
    contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # frame_ct = cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    retval, mask_thresh = cv.threshold(fgMask, 100, 255, cv.THRESH_BINARY)
    
    # set the kernal
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv.morphologyEx(mask_thresh, cv.MORPH_OPEN, kernel)
    
    min_contour_area = 100  # Define your minimum area threshold
    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_contour_area]
    
    frame_out = frame.copy()
    for cnt in large_contours:
        x, y, w, h = cv.boundingRect(cnt)
        frame_out = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
    
    # Display the resulting frame
    cv.imshow('Test', mask_thresh)    
    cv.imshow('Final frame', frame_out)    
    
    if cv.waitKey(25) & 0XFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


