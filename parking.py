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

        self.model.to('cpu')  #

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
        self.T = 20  # Tempo inicial em segundos antes de começar a incrementar o índice
        self.wc = 0.5  # Fator de crescimento do índice de suspeita para carros
        self.wi = 0.5  # Peso para fator de ocupação das vagas de estacionamento

        self.t_pedestre = 30  # Limiar de tempo para pedestres (em segundos)
        self.w_pedestre = 1.0  # Peso para intensificar o crescimento do índice de suspeita
        self.w_proximo = 1.2  # Fator para aumentar o índice de suspeita quando há interseção significativa

        # Inicializa dicionários separados para pedestres e veículos
        self.vehicle_timers = {}  # Rastreamento de tempo para veículos
        self.vehicle_indices = {}  # Índices de suspeita para veículos
        self.pedestrian_timers = {}  # Rastreamento de tempo para pedestres
        self.pedestrian_indices = {}  # Índices de suspeita para pedestres


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
        start_total = time.time()
        now = time.time()
        self.extract_tracks(im0)

        results = self.model(im0)[0]
        start_pos = time.time()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        parking_regions = [r for r in self.json if "points" in r]
        free_area_region = next((r for r in self.json if "free_area" in r), None)
        free_area = np.array(free_area_region["free_area"], np.int32).reshape((-1, 1, 2)) if free_area_region else None

        # Separar veículos e pedestres
        vehicle_boxes, vehicle_scores = [], []
        pedestrian_boxes, pedestrian_scores = [], []
        for box, score, cls in zip(boxes, scores, classes):
            box = list(map(int, box))
            if int(cls) in [3, 4, 5, 9]:
                vehicle_boxes.append(box)
                vehicle_scores.append(score)
            elif int(cls) in [0, 1]:
                pedestrian_boxes.append(box)
                pedestrian_scores.append(score)

        def stack_boxes(b, s):
            return np.column_stack((b, s)) if b else np.empty((0, 5))

        vehicle_boxes = stack_boxes(vehicle_boxes, vehicle_scores)
        pedestrian_boxes = stack_boxes(pedestrian_boxes, pedestrian_scores)

        vehicle_tracks = self.vehicle_tracker.update(vehicle_boxes, im0)
        pedestrian_tracks = self.pedestrian_tracker.update(pedestrian_boxes, im0)

        es = fs = 0
        # Verifica ocupação das vagas
        for region in self.json:
            if "points" in region:
                region_polygon = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))

                # Verifica se a vaga está ocupada
                is_occupied = any(
                    self.is_in_parking_spot([int(x1), int(y1), int(x2), int(y2)], region["points"])
                    for x1, y1, x2, y2, _ in vehicle_tracks
                )

                if is_occupied:
                    fs += 1
                else:
                    es += 1

            elif "free_area" in region:
                free_area = np.array(region["free_area"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(im0, [free_area], isClosed=True, color=self.arc, thickness=1)


        self.pr_info.update({"Occupancy": fs, "Available": es})
        i = self.pr_info["Available"] / len(parking_regions) if parking_regions else 0

        cars_in_free_area = 0
        for track in vehicle_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            box = [x1, y1, x2, y2]
            is_free = free_area is not None and cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0
            is_parked = any(self.is_in_parking_spot(box, r["points"]) for r in parking_regions)

            color = self.vehicle_parked_color if is_parked else self.vehicle_color
            label = f"Parked - {track_id}" if is_parked else f"Vehicle {track_id}"
            if not is_parked:
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
            else:
                cv2.circle(im0, (xc, yc), radius=5, color=color, thickness=-1)

            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if is_free:
                cars_in_free_area += 1
                if track_id not in self.vehicle_timers:
                    self.vehicle_timers[track_id] = now
                    self.vehicle_indices[track_id] = 0

                elapsed = now - self.vehicle_timers[track_id]
                c = max(0, (elapsed - self.T) / self.T)
                self.vehicle_indices[track_id] = c
                suspect_index = min(1, (c * self.wc) * i)

                label_s = f"Suspicion: {suspect_index:.2f} - time: {elapsed:.1f}s"
                cv2.putText(im0, label_s, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Texto colorido abaixo
                if suspect_index > 0.9:
                    text, color_s = "Suspicious", (0, 0, 255)
                elif suspect_index > 0.7:
                    text, color_s = "Reasonably suspicious", (0, 165, 255)
                elif suspect_index > 0.4:
                    text, color_s = "Slightly suspicious", (0, 255, 255)
                else:
                    text = None

                if text:
                    cv2.putText(im0, text, (x1 + 5, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_s, 2)

        for track in pedestrian_tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            color = self.pedestrian_color

            if track_id not in self.pedestrian_timers:
                self.pedestrian_timers[track_id] = now
                self.pedestrian_indices[track_id] = 0

            elapsed = now - self.pedestrian_timers[track_id]
            index = ((elapsed - self.t_pedestre) / self.t_pedestre) * self.w_pedestre if elapsed > self.t_pedestre else 0

            for vx1, vy1, vx2, vy2, _ in vehicle_tracks:
                inter_area = max(0, min(x2, vx2) - max(x1, vx1)) * max(0, min(y2, vy2) - max(y1, vy1))
                area = (x2 - x1) * (y2 - y1)
                if area > 0 and inter_area / area > 0.4:
                    index *= self.w_proximo

            self.pedestrian_indices[track_id] = min(index, 1)
            label = f"Pedestrian {track_id} - Suspicion: {index:.2f} - time: {elapsed:.1f}s"
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, 1)
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        overlay = im0.copy()
        info_text = f"Occupied: {self.pr_info['Occupancy']} | Available: {self.pr_info['Available']}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        bg_end = (10 + text_size[0] + 20, 30 + text_size[1])
        cv2.rectangle(overlay, (10, 10), bg_end, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)
        cv2.putText(im0, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 179, 255), 2)

        if free_area is not None:
            overlay = im0.copy()
            text = f"Cars in Free Area: {cars_in_free_area}"
            tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            bg_end = (10 + tsize[0] + 20, 60 + tsize[1])
            cv2.rectangle(overlay, (10, 40), bg_end, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, im0, 0.3, 0, im0)
            cv2.putText(im0, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (153, 255, 102), 2)

        print(f"Tempo de pos-processamento: {(time.time() - start_pos) * 1000:.2f} ms")
        print(f"Tempo total de processamento do frame: {(time.time() - start_total) * 1000:.2f} ms")
        return im0
