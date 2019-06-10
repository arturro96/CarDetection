import cv2

car_cascade = cv2.CascadeClassifier('russianplate.xml')
cap = cv2.VideoCapture('plates.mp4')

while True:

    ret, frame = cap.read()

    if ret:

        cars = car_cascade.detectMultiScale(frame, 1.1, 2)
        for (x, y, w, h) in cars:
            sub_img = frame[y:y + h, x:x + w]
            sub_img = cv2.GaussianBlur(sub_img, (23, 23), 30)
            frame[y:y + sub_img.shape[0], x:x + sub_img.shape[1]] = sub_img

        cv2.imshow("Result", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

