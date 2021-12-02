import cv2

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    # for making the project detect face
img = cv2.VideoCapture("face_detection.jpg")         # provide the image from which you wanna detect face
                            # or you can mention the webcam code here 0 or 1 according to your webcam

# now that as we have given the image to the code
# we need to write the code for our project to read that image
# now undersatnd
# whenyou make the project to read your image, it is gonna return you two outputs
# 1. true or false (i.e. : wheather it have read your image or not !)
# 2. it is gonna return the dimentions of image
# that's why you need to provide two variables for storing the results

result, dimentions = img.read() 

# remember
# haar cascade is trained for grey scale images only
# soo if you want your normal image to get processed by haar cascade it need to be converted into grey scale image

grey = cv2.cvtColor(dimentions, cv2.COLOR_BGR2GRAY)
# here two parameters for cvtColor are (dimentions, cv2.COLOR_BGR2GRAY) 
# so "grey" store the grey scale image

# now the next step is to detect the faces
face = detect.detectMultiScale(grey, 1.3, 5)   
# try seeing results by changing these numbers

for (x, y, w, h) in face:
    cv2.rectangle(dimentions, (x ,y), (x+w, y+h), (255, 255, 0), 2)

# let's understand the above code
# the code is for getting the square box around the face after detecting it
# you can get the rectangle by cv2.rectangle() 
# but rectangle() have 5 arguments
# 1. dimentions of the image returned by read() function
# 2. (x,y) co-ordinates (co-ordinates of bottom left)
# 3. (x+w, y+h) co-ordinated (co-ordinates of top right)
# 4. color code (for displaying the color of the box around face)
# 5. thickness of the border of the box

# let's try showing the image by following code
cv2.imshow("Demo image", dimentions)    
# "Demo image" is the name which will display on the window when image is displayed
# and second parameter is the dimntions of the image
cv2.waitKey(0)  # keep it 0, as the image will get close when you close it manually
img.release()
cv2.destroyAllWindows()  # for  closing the window

# remember
# above three statements are compulsory when you write a code to show your image