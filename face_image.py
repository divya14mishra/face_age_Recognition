import cv2
import pafy
import shutil
import matplotlib.pyplot as plt


url = 'https://www.youtube.com/watch?v=qLNhVC296YI'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture(0)
cap.set(3, 480)  # set width of the frame
cap.set(4, 640)  # set height of the frame

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']


def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('C:\\Users\\Hp\\PycharmProjects\\Face_recogination\\deploy_age.prototxt',
                                       'C:\\Users\\Hp\\PycharmProjects\\Face_recogination\\age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('C:\\Users\\Hp\\PycharmProjects\\Face_recogination\\deploy_gender.prototxt',
                                          'C:\\Users\\Hp\\PycharmProjects\\Face_recogination\\gender_net.caffemodel')
    return age_net, gender_net


def video_detector(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0
    while True:
        ret, image = cap.read()
        # print("IMAGE : ", image)
        face_cascade = cv2.CascadeClassifier('C:\\Users\\Hp\\haarcascade_frontalface_alt.xml')
        imgofface = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if (len(faces) > 0):
            print("Found {} faces".format(str(len(faces))))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Get Face
            face_img = image[y:y + h, x:x + w].copy()
            # cv2.imshow("face_boundry", face_img)

            imgofface = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            count += 1
            # print("FACE OF PERSON", face_img)

            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # print("Gender : " + gender)
            # Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            # print("Age Range: " + age)
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        plt.figure(figsize=(6, 6), dpi=80)
        plt.imshow(imgofface)
        plt.title("Face Image")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('C:\\Users\\Hp\\PycharmProjects\\face_pictures\\face_read{}.png'.format(count))
        # plt.show()
       

        cv2.imshow('frame', image)

    # 0xFF is a hexadecimal constant which is 11111111 in binary.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

# cv2.destroyAllWindows()
if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
    shutil.make_archive("face_captures_folder", "zip", "C:\\Users\\Hp\\PycharmProjects\\face_pictures")
