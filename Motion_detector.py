import cv2,pandas
from datetime import datetime
status_list = [None,None]
motion_time = []
first_frame = None
video = cv2.VideoCapture(0)
df = pandas.DataFrame(columns=["Start","Exit"])
while True:
    status = 0
    check, frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None :
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_frame = cv2.threshold(delta_frame,60,255,cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame,None,iterations=2)

    (_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 20000 :
            continue
        status = 1
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
        motion_time.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        motion_time.append(datetime.now())

    cv2.imshow("Video",gray)
    cv2.imshow("delta_frame",delta_frame)
    cv2.imshow("Thrshold_frame",thresh_frame)
    cv2.imshow("Actual Frame",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            motion_time.append(datetime.now())

        break

for i in range(0,len(motion_time),2):
    df = df.append({"Start":motion_time[i],"Exit":motion_time[i+1]},ignore_index=True)
df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
