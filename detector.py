import cv2

face_cascade = cv2.CascadeClassifier('cars.xml')
cap = cv2.VideoCapture('video1.avi')

while True:

    ret, frame = cap.read()

    if ret:

        cars = face_cascade.detectMultiScale(frame, 1.1, 2)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Result", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
