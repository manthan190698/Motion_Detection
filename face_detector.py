import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("DSCN0959.jpg")
gry_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

face=face_cascade.detectMultiScale(gry_img,
scaleFactor=1.2,
minNeighbors=5)

for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

print(type(face))
print(face)

resized=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))

cv2.imshow("Image",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
