# in the file "script.py" we have seen how how to detect the face from the single image 
# but what if, you wanna detect the faces of from all the images present in the same folder ?
# sooo let's see how to do that

# we need to use something called "glob"
# which is the module use to detect all the files from the same folder

import cv2, glob

all_images = glob.glob("*.jpg")
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for image in all_images:
    img = cv2.imread(image)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect.detectMultiScale(grey_img, 1.3, 5)

    for (x, y, w, h) in faces:
        final_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 255, 0), 2)

    cv2.imshow("Final window", final_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()