import cv2
import face_recognition_models.models
# if we pass 0 as 2nd parameter it will convert image in greyscale
# if we pass 1 as 2nd parameter it will convert image in colored image
img = cv2.imread("E:\\sample2.jpg", 1)
resized = cv2.resize(img, (400, 500)) # it will resize the image


# resize/d_new = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
# resized_new = cv2.resize(img, (int(img.shape[1]*3), int(img.shape[0]*3)))
# as image colored image it will print 3d array
# if it is black & white aur greyscale image it will print 2d array

print("3D image ")
print(img)
print("type of array :", type(img))
print("shape or size of array ", img.shape)

face_cascade = cv2.CascadeClassifier("C:\\Users\Hp\\PycharmProjects\\Face_recogination\\haarcascade_frontalface_default.xml")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.5, minNeighbors=2)
print(type(faces))
print(faces)
for x,y,w,h in faces:
  resized  = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# cv2.imshow("MISHKA", img)
cv2.imshow("MISHKA", resized)
cv2.waitKey(0)


cv2.destroyAllWindows()

