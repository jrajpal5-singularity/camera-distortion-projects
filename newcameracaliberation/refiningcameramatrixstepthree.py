# Refining the camera matrix using parameters obtained by calibration
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Method 1 to undistort the image
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Method 2 to undistort the image
mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)

dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# Displaying the undistorted image
cv2.imshow("undistorted image",dst)
cv2.waitKey(0)