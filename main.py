import cv2
from vidgear.gears import CamGear

from parking import ParkingManagement

stream = CamGear(source='https://www.youtube.com/watch?v=EPKWu223XEg', stream_mode = True, logging=True).start()





# Initialize parking management object
parking_manager =  ParkingManagement(
    model="yolo11n.pt",# path to model file
    classes=[2],
    json_file="bounding_boxes.json",  # path to parking annotations file
)
count=0
while True:
    im0 = stream.read()
    count += 1
    if count % 2 != 0:
        continue
    im0=cv2.resize(im0,(1020,500))
    im0 = parking_manager.process_data(im0)
    cv2.imshow("im0",im0)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()