import cv2 
from yolo_segmentation import YOLOSegmentation

# Segmentation detector
ys = YOLOSegmentation("yolov8m-seg.pt")
names = ys.model.names

# Open the capture device
capture = cv2.VideoCapture(0)

# Lower the resolution of the frames to reduce processing load
target_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
target_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    # Resize the frame to the target dimensions
    resized = cv2.resize(frame, (target_width, target_height))

    bboxes, classes, segmentations, scores = ys.detect(resized)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        (x, y, x2, y2) = bbox
        if(names[class_id] == 'bottle'):

            cv2.rectangle(resized, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.polylines(resized, [seg], True, (255, 0, 0), 2)
            cv2.putText(resized, str(names[class_id]), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("image", resized)

    # Throttle the frame rate to reduce processing load
    if cv2.waitKey(100) & 0xFF == ord('d'):
        break

# Release the capture and close the OpenCV windows
capture.release()
cv2.destroyAllWindows()
