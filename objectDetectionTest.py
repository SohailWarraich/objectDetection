import cv2

Conf_threshold = 0.4
NMS_threshold = 0.4

class_name = []
with open('classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
    # print(class_name)

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

cap = cv2.VideoCapture('output.avi')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

    for (classid, score, box) in zip(classes, scores, boxes):
        if classid == 2:
            print(classid)
            label = "%s : %.2f" % (class_name[classid], round(score, 2))  # Round to 2 decimal places
            cv2.rectangle(frame, box, (255, 255, 255), 1)  # Green bounding box
            cv2.putText(frame, label, (box[0], box[1]+10),
            cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)  # Green text
            cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
