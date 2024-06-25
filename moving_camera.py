'''
Motion detection for moving camera

'''
import numpy as np
import cv2

video_path = "cars.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video not loaded")
    exit()

backSub = cv2.createBackgroundSubtractorMOG2()

# ORB detector
orb = cv2.ORB_create()
    
# Get the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    
while True:
    ret, frame = cap.read()
    if not ret:
        print("video capture ended")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 10.0)
    
    # Warp current frame
    height, width = frame.shape[:2]
    stabilized_frame = cv2.warpPerspective(frame, H, (width, height))
    
    # Apply background subtraction on stabilized frame
    fgMask = backSub.apply(stabilized_frame)
    
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 200:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(stabilized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the original frame, stabilized frame, and the foreground mask
    # cv2.imshow('Original Frame', frame)
    cv2.imshow('Stabilized Frame', stabilized_frame)
    cv2.imshow('Foreground Mask', fgMask)
    
    # Update the previous frame and previous gray frame
    prev_frame = frame.copy()
    prev_gray = gray.copy()
    
    # Press 'q' to exit the video
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()


