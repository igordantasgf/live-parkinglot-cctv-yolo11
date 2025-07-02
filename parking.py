import json
import time  # Importa o módulo para rastrear o tempo

import cv2
import numpy as np

from ultralytics.solutions.solutions import LOGGER, BaseSolution, check_requirements
from ultralytics.utils.plotting import Annotator
from deep_sort import DeepSORT


class ParkingPtsSelection:
    """
    A class for selecting and managing parking zone points on images using a Tkinter-based UI.

    This class provides functionality to upload an image, select points to define parking zones, and save the
    selected points to a JSON file. It uses Tkinter for the graphical user interface.

    Attributes:
        tk (module): The Tkinter module for GUI operations.
        filedialog (module): Tkinter's filedialog module for file selection operations.
        messagebox (module): Tkinter's messagebox module for displaying message boxes.
        master (tk.Tk): The main Tkinter window.
        canvas (tk.Canvas): The canvas widget for displaying the image and drawing bounding boxes.
        image (PIL.Image.Image): The uploaded image.
        canvas_image (ImageTk.PhotoImage): The image displayed on the canvas.
        rg_data (List[List[Tuple[int, int]]]): List of bounding boxes, each defined by 4 points.
        current_box (List[Tuple[int, int]]): Temporary storage for the points of the current bounding box.
        imgw (int): Original width of the uploaded image.
        imgh (int): Original height of the uploaded image.
        canvas_max_width (int): Maximum width of the canvas.
        canvas_max_height (int): Maximum height of the canvas.

    Methods:
        setup_ui: Sets up the Tkinter UI components.
        initialize_properties: Initializes the necessary properties.
        upload_image: Uploads an image, resizes it to fit the canvas, and displays it.
        on_canvas_click: Handles mouse clicks to add points for bounding boxes.
        draw_box: Draws a bounding box on the canvas.
        remove_last_bounding_box: Removes the last bounding box and redraws the canvas.
        redraw_canvas: Redraws the canvas with the image and all bounding boxes.
        save_to_json: Saves the bounding boxes to a JSON file.

    Examples:
        >>> parking_selector = ParkingPtsSelection()
        >>> # Use the GUI to upload an image, select parking zones, and save the data
    """

    def __init__(self):
        """Initializes the ParkingPtsSelection class, setting up UI and properties for parking zone point selection."""
        check_requirements("tkinter")
        import tkinter as tk
        from tkinter import filedialog, messagebox

        self.tk, self.filedialog, self.messagebox = tk, filedialog, messagebox
        self.setup_ui()
        self.initialize_properties()
        self.master.mainloop()

    def setup_ui(self):
        """Sets up the Tkinter UI components for the parking zone points selection interface."""
        self.master = self.tk.Tk()
        self.master.title("Ultralytics Parking Zones Points Selector")
        self.master.resizable(False, False)

        # Canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")
        self.canvas.pack(side=self.tk.BOTTOM)

        # Button frame with buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        for text, cmd in [
            ("Upload Image", self.upload_image),
            ("Remove Last BBox", self.remove_last_bounding_box),
            ("Save", self.save_to_json),
        ]:
            self.tk.Button(button_frame, text=text, command=cmd).pack(side=self.tk.LEFT)

    def initialize_properties(self):
        """Initialize properties for image, canvas, bounding boxes, and dimensions."""
        self.image = self.canvas_image = None
        self.rg_data, self.current_box = [], []
        self.imgw = self.imgh = 0
        self.canvas_max_width, self.canvas_max_height = 1020, 500

    def upload_image(self):
        """Uploads and displays an image on the canvas, resizing it to fit within specified dimensions."""
        from PIL import Image, ImageTk  # scope because ImageTk requires tkinter package

        self.image = Image.open(self.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")]))
        if not self.image:
            return

        self.imgw, self.imgh = self.image.size
        aspect_ratio = self.imgw / self.imgh
        canvas_width = (
            min(self.canvas_max_width, self.imgw) if aspect_ratio > 1 else int(self.canvas_max_height * aspect_ratio)
        )
        canvas_height = (
            min(self.canvas_max_height, self.imgh) if aspect_ratio <= 1 else int(canvas_width / aspect_ratio)
        )

        self.canvas.config(width=canvas_width, height=canvas_height)
        self.canvas_image = ImageTk.PhotoImage(self.image.resize((canvas_width, canvas_height), Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.rg_data.clear(), self.current_box.clear()

    def on_canvas_click(self, event):
        """Handles mouse clicks to add points for bounding boxes on the canvas."""
        self.current_box.append((event.x, event.y))
        self.canvas.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
        if len(self.current_box) == 4:
            self.rg_data.append(self.current_box.copy())
            self.draw_box(self.current_box)
            self.current_box.clear()

    def draw_box(self, box):
        """Draws a bounding box on the canvas using the provided coordinates."""
        for i in range(4):
            self.canvas.create_line(box[i], box[(i + 1) % 4], fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Removes the last bounding box from the list and redraws the canvas."""
        if not self.rg_data:
            self.messagebox.showwarning("Warning", "No bounding boxes to remove.")
            return
        self.rg_data.pop()
        self.redraw_canvas()

    def redraw_canvas(self):
        """Redraws the canvas with the image and all bounding boxes."""
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
        for box in self.rg_data:
            self.draw_box(box)

    def save_to_json(self):
        """Saves the selected parking zone points to a JSON file with scaled coordinates."""
        scale_w, scale_h = self.imgw / self.canvas.winfo_width(), self.imgh / self.canvas.winfo_height()
        data = [{"points": [(int(x * scale_w), int(y * scale_h)) for x, y in box]} for box in self.rg_data]
        with open("bounding_boxes.json", "w") as f:
            json.dump(data, f, indent=4)
        self.messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")
















class ParkingManagement(BaseSolution):
    """
    Manages parking occupancy and availability using YOLO model for real-time monitoring and visualization.
    """

    def __init__(self, **kwargs):
        """Initializes the parking management system with a YOLO model and visualization settings."""
        super().__init__(**kwargs)

        self.json_file = self.CFG["json_file"]  # Load JSON data
        if self.json_file is None:
            LOGGER.warning("❌ json_file argument missing. Parking region details required.")
            raise ValueError("❌ Json file path can not be empty")

        with open(self.json_file) as f:
            self.json = json.load(f)

        self.pr_info = {"Occupancy": 0, "Available": 0}  # dictionary for parking information

        self.arc = (0, 255, 0)  # available region color
        self.occ = (0, 0, 255)  # occupied region color
        self.dc = (255, 0, 189)  # centroid color for each box
        
        # Inicializa o DeepSORT para tracking de pedestres e veículos
        self.pedestrian_tracker = DeepSORT(max_age=30, min_hits=1, iou_threshold=0.5)
        self.vehicle_tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.5)
        
        # Cores para bounding boxes
        self.pedestrian_color = (0, 255, 255)  # Amarelo
        self.pedestrian_danger_color = (255, 192, 203)  # Rosa
        self.vehicle_color = (255, 0, 0)  # Azul
        self.vehicle_parked_color = (0, 0, 255)  # Vermelho

        self.car_timers = {}  # Dicionário para rastrear o tempo de cada veículo na free_area
        self.car_moving_indices = {}  # Dicionário para armazenar o índice de cada veículo
        self.T = 10  # Tempo inicial em segundos antes de começar a incrementar o índice
        self.wc = 0.1  # Fator de crescimento do índice de suspeita para carros
        self.wi = 0.5  # Peso para fator de ocupação das vagas de estacionamento


    def check_intersection(self, ped_box, car_box):
        """
        Verifica se há interseção entre a bounding box do pedestre e do veículo
        """
        x1 = max(ped_box[0], car_box[0])
        y1 = max(ped_box[1], car_box[1])
        x2 = min(ped_box[2], car_box[2])
        y2 = min(ped_box[3], car_box[3])
        
        if x1 < x2 and y1 < y2:
            return True
        return False

    def is_in_parking_spot(self, box, region_points):
        """
        Verifica se um veículo está em uma vaga de estacionamento
        """
        x1, y1, x2, y2 = box
        xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)
        pts_array = np.array(region_points, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.pointPolygonTest(pts_array, (xc, yc), False) >= 0

    def process_data(self, im0):
        """
        Processa os dados do modelo para gerenciar o estacionamento.

        Esta função analisa a imagem de entrada, extrai as trilhas e determina o status de ocupação das regiões de
        estacionamento definidas no arquivo JSON. Também verifica se os carros detectados estão dentro da free_area.

        Args:
            im0 (np.ndarray): A imagem de entrada para inferência.

        Returns:
            np.ndarray: A imagem anotada.
        """
        self.extract_tracks(im0)  # Realiza a inferência e extrai as detecções
        es, fs = 0, 0  # Vagas disponíveis e ocupadas
        annotator = Annotator(im0, self.line_width)  # Inicializa o anotador
        
        # Carrega a free_area do JSON
        free_area = None
        for region in self.json:
            if "free_area" in region:
                free_area = np.array(region["free_area"], dtype=np.int32).reshape((-1, 1, 2))
                break

        cars_in_free_area = 0  # Contador para carros na free_area
        
        # Separa detecções de veículos e pedestres
        vehicle_boxes = []
        pedestrian_boxes = []
        vehicle_scores = []
        pedestrian_scores = []
        
        # Extrai as detecções do modelo
        results = self.model(im0)[0]  # Obtém as detecções do modelo YOLO
        
        # Extrai boxes, scores e classes
        boxes = results.boxes.xyxy.cpu().numpy()  # Coordenadas das bounding boxes
        scores = results.boxes.conf.cpu().numpy()  # Confidências
        classes = results.boxes.cls.cpu().numpy()  # Classes
        
        # Processa cada detecção
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            if cls in [3, 4, 5, 9]:  # Classes de veículos: car, van, truck, motor
                vehicle_boxes.append([x1, y1, x2, y2])
                vehicle_scores.append(score)
            elif cls in [0, 1]:  # Classes de pedestres: pedestrian, people
                pedestrian_boxes.append([x1, y1, x2, y2])
                pedestrian_scores.append(score)

        # Converte para numpy arrays
        vehicle_boxes = np.array(vehicle_boxes)
        pedestrian_boxes = np.array(pedestrian_boxes)
        vehicle_scores = np.array(vehicle_scores)
        pedestrian_scores = np.array(pedestrian_scores)

        # Adiciona scores às bounding boxes
        if len(vehicle_boxes) > 0:
            vehicle_boxes = np.column_stack((vehicle_boxes, vehicle_scores))
        if len(pedestrian_boxes) > 0:
            pedestrian_boxes = np.column_stack((pedestrian_boxes, pedestrian_scores))

        # Atualiza os trackers
        vehicle_tracks = self.vehicle_tracker.update(vehicle_boxes, im0)
        pedestrian_tracks = self.pedestrian_tracker.update(pedestrian_boxes, im0)

        # Desenha as demarcações das vagas e verifica ocupação
        for region in self.json:
            if "points" in region:
                region_polygon = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
                # cv2.polylines(im0, [region_polygon], isClosed=True, color=self.arc, thickness=2)

                # Verifica se a vaga está ocupada
                is_occupied = False
                for track in vehicle_tracks:
                    x1, y1, x2, y2, track_id = track
                    if self.is_in_parking_spot([x1, y1, x2, y2], region["points"]):
                        is_occupied = True
                        break

                if is_occupied:
                    fs += 1  # Incrementa vagas ocupadas
                else:
                    es += 1  # Incrementa vagas disponíveis

            if "free_area" in region:
                region_polygon = np.array(region["free_area"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(im0, [region_polygon], isClosed=True, color=self.arc, thickness=1)

        # Atualiza informações de estacionamento
        self.pr_info["Occupancy"] = fs
        self.pr_info["Available"] = es

        # Desenha as bounding boxes dos veículos com IDs únicos
        for track in vehicle_tracks:
            is_in_free_area = False
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Verifica se o veículo está na free_area
            if free_area is not None and cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0:
                is_in_free_area = True
                cars_in_free_area += 1

            # Verifica se o veículo está em uma vaga
            is_parked = False
            for region in self.json:
                if "points" in region and self.is_in_parking_spot([x1, y1, x2, y2], region["points"]):
                    is_parked = True
                    break
            
            if is_parked:
                color = self.vehicle_parked_color
                label = f"Parked - {track_id}"
                cv2.circle(im0, (xc, yc), radius=5, color=color, thickness=-1)
                cv2.putText(im0, label, (xc + 10, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif is_in_free_area:
                color = self.vehicle_color
                label = f"Vehicle {track_id}"
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                cv2.putText(im0, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Desenha as bounding boxes dos pedestres com IDs únicos
        for track in pedestrian_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            color = self.pedestrian_color
            label = f"Pedestrian {track_id}"
            
            # Desenha a bounding box e o label
            overlay = im0.copy()  # Reinicializa o overlay antes de cada operação
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_width, text_height = text_size[0], text_size[1]
            background_start = (x1, y1 - text_height - 10)
            background_end = (x1 + text_width + 10, y1 - 5)

            cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)  # Desenha o retângulo preto
            cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)  # Aplica opacidade ao fundo

            # Adiciona o texto sobre o fundo
            cv2.putText(im0, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Desenha o centro do pedestre e uma linha indicando a direção do movimento
            cv2.circle(im0, (xc, yc), radius=3, color=color, thickness=-1)
            
        # Exibe informações de estacionamento
        overlay = im0.copy()
        text_size = cv2.getTextSize(f"Occupied: {self.pr_info['Occupancy']}, Available: {self.pr_info['Available']}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_width, text_height = text_size[0], text_size[1]
        background_start = (10, 10)
        background_end = ( text_width, 20 + text_height*2)
        cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)
        cv2.putText(
            im0,
            f"Occupied: {self.pr_info['Occupancy']}, Available: {self.pr_info['Available']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (128, 179, 255),
            2,
            cv2.LINE_AA,
        )

        # Exibe informações da free_area com fundo preto opaco
        if free_area is not None:
            overlay = im0.copy()
            text_size = cv2.getTextSize(f"Cars in Free Area: {cars_in_free_area}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_width, text_height = text_size[0], text_size[1]
            background_start = (10, 40)
            background_end = (10 + text_width + 20, 60 + text_height)
            # cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)
            cv2.putText(
                im0,
                f"Cars in Free Area: {cars_in_free_area}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (153, 255, 102),
                2,
                cv2.LINE_AA,
            )

        # Inicializa variáveis para o cálculo do índice de suspeita
        total_vagas = len([region for region in self.json if "points" in region])  # Número total de vagas
        i = self.pr_info["Available"] / total_vagas  # Calcula o fator "i" baseado nas vagas disponíveis
        suspect_indices = []  # Lista para armazenar os índices de suspeita e IDs dos veículos
        w1 = 0.5 # expoente para controlar curvatura de crescimento do índice de suspeita

        # Atualiza o loop para veículos na free_area
        for track in vehicle_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Verifica se o veículo está na free_area
            if free_area is not None and cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0:
                cars_in_free_area += 1

                # Calcula o índice de movimentação "i" para o veículo
                if track_id not in self.car_timers:
                    self.car_timers[track_id] = time.time()  # Inicializa o tempo de entrada na free_area
                    self.car_moving_indices[track_id] = 0  # Inicializa o índice de movimentação

                elapsed_time = (time.time() - self.car_timers[track_id])
                if elapsed_time > self.T:  # Apenas incrementa o índice após ultrapassar o limiar T
                    self.car_moving_indices[track_id] = (elapsed_time - self.T) / self.T  # Normaliza o índice de movimentação
                else:
                    self.car_moving_indices[track_id] = 0

                c = self.car_moving_indices[track_id]

                # Calcula o índice de suspeita
                if (elapsed_time < self.T):
                    suspect_index = c
                    suspect_indices.append((track_id, suspect_index))
                else: # t >= T
                    # ------------------------------------------------------
                    #       Fórmula para calcular o índice de suspeita
                    # ------------------------------------------------------
                    #
                    suspect_index = min(1, (c *  w1) * i )
                    # suspect_index = min(1, c * (self.wc + i * self.wi)/ self.wc + self.wi)
                    #
                    # ------------------------------------------------------
                    suspect_indices.append((track_id, suspect_index))

                # Desenha a bounding box do veículo com o índice de suspeita
                color = self.vehicle_color
                label = f"Vehicle {track_id} - Suspicion: {suspect_index:.2f} - time: {elapsed_time:.2f}s"

                # Adiciona o fundo preto opaco acima do bounding box
                overlay = im0.copy()  # Reinicializa o overlay antes de cada operação

                # Calcula o tamanho do texto
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_width, text_height = text_size[0], text_size[1]
                background_start = (x1, y1 - text_height - 10)
                background_end = (x1 + text_width + 10, y1 - 5)

                # Adiciona o fundo preto opaco acima do bounding box
                cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)

                # Adiciona o texto sobre o fundo
                cv2.putText(im0, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Define o texto e a cor com base no nível de suspeita
                if suspect_index > 0.9:
                    suspicion_text = "Suspicious"
                    suspicion_color = (0, 0, 255)  # Vermelho
                elif suspect_index > 0.7:
                    suspicion_text = "Reasonably suspicious"
                    suspicion_color = (0, 165, 255)  # Laranja
                elif suspect_index > 0.4:
                    suspicion_text = "Slightly suspicious"
                    suspicion_color = (0, 255, 255)  # Amarelo
                else:
                    suspicion_text = None

                # Exibe o texto abaixo do bounding box, no lado esquerdo
                if suspicion_text:
                    overlay = im0.copy()  # Reinicializa o overlay antes de desenhar o fundo do texto
                    text_size = cv2.getTextSize(suspicion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    text_width, text_height = text_size[0], text_size[1]
                    background_start = (x1, y2 + 10)
                    background_end = (x1 + text_width + 10, y2 + 10 + text_height + 10)

                    # Adiciona o fundo preto opaco
                    cv2.rectangle(overlay, background_start, background_end, (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)

                    # Adiciona o texto sobre o fundo
                    cv2.putText(im0, suspicion_text, (x1 + 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, suspicion_color, 2)

        overlay = im0.copy()

        # # Exibe a tabela dinâmica no canto superior direito com fundo cinza opaco
        # start_x, start_y = im0.shape[1] - 350, 30  # Posição inicial da tabela
        # table_width, table_height = 280, 30 * (len(suspect_indices) + 1)  # Dimensões da tabela
        # background_color = (50, 50, 50)  # Cor do fundo (cinza escuro)
        # opacity = 0.7  # Opacidade do fundo

        # cv2.rectangle(overlay, (start_x, start_y), (start_x + table_width, start_y + table_height), background_color, -1)
        # cv2.addWeighted(overlay, opacity, im0, 1 - opacity, 0, im0)

        # # Adiciona o título da tabela
        # cv2.putText(im0, "Suspicion Table", (start_x + 10, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # for idx, (track_id, suspect_index) in enumerate(suspect_indices):
        #     text = f"ID: {track_id} | Suspicion: {suspect_index:.2f} - c: {c:.2f}, i: {i:.2f}"
        #     cv2.putText(im0, text, (start_x, start_y + 30 * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return im0