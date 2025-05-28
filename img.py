import cv2    
import time
from vidgear.gears import CamGear

cpt = 0
maxFrames = 1 # if you want 5 frames only.
video_path = "ufba_odonto.mp4"  # Substitua pelo caminho do vídeo se necessário

if video_path:
    stream = cv2.VideoCapture("videos/"+video_path)  # Substitua pelo caminho do vídeo se necessário
else:
    stream = CamGear(source='https://www.youtube.com/watch?v=EPKWu223XEg', stream_mode = True, logging=True).start()

if video_path:
    ret, frame = stream.read()
    if ret:
        frame = cv2.resize(frame, (1020, 500))
        # Aqui você pode adicionar a lógica para definir os bounding boxes no frame
        cv2.imshow("Definir Bounding Boxes", frame)
        cv2.imwrite(f"bboxes_images/img_{video_path}.jpg", frame)
        cv2.waitKey(0)  # Aguarda o usuário fechar a janela
        cv2.destroyAllWindows()
else:
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