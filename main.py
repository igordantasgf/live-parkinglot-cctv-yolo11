import cv2
from vidgear.gears import CamGear
import numpy as np

from parking import ParkingManagement

video_path = "ufba_odonto.mp4"  # Substitua pelo caminho do vídeo se necessário
# video_path = False
if video_path:
    stream = cv2.VideoCapture("videos/" + video_path)
else:
    stream = CamGear(source='https://www.youtube.com/watch?v=EPKWu223XEg', stream_mode=True, logging=True).start()

# Initialize parking management object
parking_manager =  ParkingManagement(
    model="yolo11s_visdrone.pt",# path to model file
    classes=[0, 2, 3, 4, 5, 9],  # 0 para pedestres, 2 para carros
    json_file=("bboxes_videos/" + video_path + ".json") if video_path else "stream_bounding_boxes.json",  # path to parking annotations file
)

def jump_to_time(stream, time_in_seconds):
    if video_path:
        fps = stream.get(cv2.CAP_PROP_FPS)  # Obtém o FPS do vídeo
        frame_number = int(fps * time_in_seconds)  # Calcula o número do frame correspondente
        stream.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Redireciona para o frame
        print(f"Redirecionado para {time_in_seconds} segundos.")

count = 0
i = 1
while True:
    if video_path:
        ret, im0 = stream.read()
        if not ret:
            break
    else:
        im0 = stream.read()

    count += 1
    if count % 2 != 0:
        continue
    im0 = cv2.resize(im0, (1020, 500))
    im0 = parking_manager.process_data(im0)
    cv2.imshow("im0", im0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('j'):  
        jump_to_time(stream, 100)  
    if key == ord('i'):  
        jump_to_time(stream, 120) 
    if key == ord('k'):  
        jump_to_time(stream, 250)  
    elif key == ord('p'):  # Salva a imagem corrente
        cv2.imwrite(f"current_frame_{i}.jpg", im0)
        i += 1
        print(f"Imagem salva como 'current_frame{i}.jpg'")
    elif key == ord('q'):
        break

if video_path:
    stream.release()
else:
    stream.stop()
cv2.destroyAllWindows()