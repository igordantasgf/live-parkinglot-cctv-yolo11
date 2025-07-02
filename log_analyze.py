import re
import statistics
import pandas as pd

# Exemplo de entrada (você deve substituir isso pela entrada real completa, se necessário)
with open("logs/analise1.log", "r") as f:
    log_text = f.read()
    print(len(log_text))  # Exibe o conteúdo do log para verificação

# Regex para capturar cada linha com detecções e tempos
pattern = re.compile(
    r"(?P<frame_id>\d+): 384x640 (?P<detections>.+?), (?P<total_time>\d+\.\d+)ms\s+Speed: (?P<preprocess>\d+\.\d+)ms preprocess, (?P<inference>\d+\.\d+)ms inference, (?P<postprocess>\d+\.\d+)ms postprocess"
)

data = []

for match in pattern.finditer(log_text):
    detections_str = match.group("detections")
    
    # Contadores de objetos
    pedestrians = cars = vans = 0
    if "pedestrian" in detections_str:
        pedestrian_match = re.search(r"(\d+) pedestrian", detections_str)
        if pedestrian_match:
            pedestrians = int(pedestrian_match.group(1))
        else:
            pedestrians = 1  # caso "1 pedestrian"
    if "car" in detections_str:
        cars = int(re.search(r"(\d+) cars?", detections_str).group(1))
    if "van" in detections_str:
        vans = int(re.search(r"(\d+) vans?", detections_str).group(1))
        if not vans:
            vans = 1

    data.append({
        "frame": int(match.group("frame_id")),
        "pedestrians": pedestrians,
        "cars": cars,
        "vans": vans,
        "total_time": float(match.group("total_time")),
        "preprocess": float(match.group("preprocess")),
        "inference": float(match.group("inference")),
        "postprocess": float(match.group("postprocess")),
    })

# Converter para DataFrame
df = pd.DataFrame(data)

# Estatísticas relevantes
summary = {
    "total_frames": len(df),
    "avg_total_time": df["total_time"].mean(),
    "std_total_time": df["total_time"].std(),
    "max_total_time": df["total_time"].max(),
    "min_total_time": df["total_time"].min(),
    "avg_preprocess": df["preprocess"].mean(),
    "avg_inference": df["inference"].mean(),
    "avg_postprocess": df["postprocess"].mean(),
    "avg_pedestrians": df["pedestrians"].mean(),
    "avg_cars": df["cars"].mean(),
    "avg_vans": df["vans"].mean(),
}
print(summary)
