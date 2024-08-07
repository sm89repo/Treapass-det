import cv2
import numpy as np
from shapely import Point, Polygon
from ultralytics import YOLO

path = r"C:\Users\Madhwanath\Downloads\Shopping, People, Commerce, Mall, Many, Crowd, Walking   Free Stock video footage   YouTube.mp4"

cap = cv2.VideoCapture(path)
model = YOLO('yolov8s.pt')
points = [(287, 559), (679, 265), (710, 294), (388, 607)]
trespass_polygon = Polygon(points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, device='cpu', classes=0, persist=True)
    boxes = results[0].boxes
    for box in boxes:
        id = box.id.numpy().astype(int)
        x1, y1, x2, y2 = box.xyxy.numpy().astype(int)[0]
        center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        person_center = Point(center_point)

        if person_center.intersects(trespass_polygon):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Trespasser', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'per:{id}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.putText(frame, 'Trespassing Area', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
