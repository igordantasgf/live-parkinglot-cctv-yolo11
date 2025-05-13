import cv2    
import time
from vidgear.gears import CamGear

cpt = 0
maxFrames = 1 # if you want 5 frames only.
stream = CamGear(source='https://www.youtube.com/watch?v=EPKWu223XEg', stream_mode = True, logging=True).start()

while cpt < maxFrames:
    frame = stream.read()
    frame=cv2.resize(frame,(1020,500))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\Igor\Documents\Material Faculdade\TCC\live-parkinglot-cctv-yolo11/img_%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
stream.stop()   
cv2.destroyAllWindows()