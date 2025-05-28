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
        self.pedestrian_tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
        self.vehicle_tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Cores para bounding boxes
        self.pedestrian_color = (0, 255, 255)  # Amarelo
        self.pedestrian_danger_color = (255, 192, 203)  # Rosa
        self.vehicle_color = (255, 0, 0)  # Azul
        self.vehicle_parked_color = (0, 0, 255)  # Vermelho

        self.car_timers = {}  # Dicionário para rastrear o tempo de cada veículo na free_area
        self.car_moving_indices = {}  # Dicionário para armazenar o índice de cada veículo
        self.T = 20  # Tempo inicial em segundos antes de começar a incrementar o índice

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
        Processes the model data for parking lot management.

        This function analyzes the input image, extracts tracks, and determines the occupancy status of parking
        regions defined in the JSON file. It also checks if detected cars are within the free_area.

        Args:
            im0 (np.ndarray): The input inference image.

        Returns:
            np.ndarray: The annotated image.
        """
        self.extract_tracks(im0)  # extract tracks from im0
        es, fs = 0, 0  # empty slots, filled slots
        annotator = Annotator(im0, self.line_width)  # init annotator

        # Load free_area from JSON
        free_area = None
        for region in self.json:
            if "free_area" in region:
                free_area = np.array(region["free_area"], dtype=np.int32).reshape((-1, 1, 2))
                break

        cars_in_free_area = 0  # Counter for cars in the free_area
        
        # Separa detecções de carros e pedestres
        car_boxes = []
        pedestrian_boxes = []
        
        for box, cls in zip(self.boxes, self.clss):
            x1, y1, x2, y2 = map(int, box)
            if cls == 2:  # Carro
                car_boxes.append([x1, y1, x2, y2])
            elif cls == 0:  # Pedestre
                pedestrian_boxes.append([x1, y1, x2, y2])

        # Atualiza o DeepSORT para veículos
        if car_boxes:
            car_boxes = np.array(car_boxes)
            tracked_vehicles = self.vehicle_tracker.update(car_boxes, [])
            
            # Desenha as bounding boxes dos veículos
            for track in tracked_vehicles:
                x1, y1, x2, y2, track_id = map(int, track)
                xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Verifica se o veículo está na free_area
                if free_area is not None and cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0:
                    if track_id not in self.car_timers:
                        self.car_timers[track_id] = time.time()  # Inicia o timer para o veículo
                        self.car_moving_indices[track_id] = 0  # Inicializa o índice do veículo
                    else:
                        elapsed_time = time.time() - self.car_timers[track_id]
                        if elapsed_time > self.T:
                            # Incrementa o índice com crescimento exponencial quadrático
                            normalized_time = (elapsed_time - self.T) / self.T
                            self.car_moving_indices[track_id] = min(1, normalized_time ** 2)
                else:
                    # Remove o veículo do dicionário se ele sair da free_area
                    if track_id in self.car_timers:
                        del self.car_timers[track_id]
                        del self.car_moving_indices[track_id]

                # Verifica se o veículo está em uma vaga
                is_parked = False
                for region in self.json:
                    if "points" in region and self.is_in_parking_spot([x1, y1, x2, y2], region["points"]):
                        is_parked = True
                        break
                
                # Desenha a bounding box apenas se o veículo não estiver estacionado
                if not is_parked:
                    # Escolhe a cor baseada no estado do veículo
                    color = self.vehicle_color
                    
                    # Desenha a bounding box
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                    
                    # Adiciona o ID do tracker
                    cv2.putText(im0, f"V{track_id}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Atualiza o DeepSORT para pedestres
        if pedestrian_boxes:
            pedestrian_boxes = np.array(pedestrian_boxes)
            tracked_pedestrians = self.pedestrian_tracker.update(pedestrian_boxes, car_boxes)
            
            # Desenha as bounding boxes dos pedestres
            for track in tracked_pedestrians:
                x1, y1, x2, y2, track_id = map(int, track)
                
                # Verifica interseção com carros
                is_intersecting = False
                for car_box in car_boxes:
                    if self.check_intersection([x1, y1, x2, y2], car_box):
                        is_intersecting = True
                        break
                
                # Escolhe a cor baseada na interseção
                color = self.pedestrian_danger_color if is_intersecting else self.pedestrian_color
                
                # Desenha a bounding box
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                
                # Adiciona o ID do tracker
                cv2.putText(im0, f"P{track_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Constantes para o cálculo do índice
        peso_car_moving = 0.8  # Peso do índice de movimento do carro
        full_park_index = 0.05

        # Lista para armazenar os índices calculados
        vehicle_indices = []

        # Atualiza o índice de cada veículo na free_area
        for track in tracked_vehicles if 'tracked_vehicles' in locals() else []:
            x1, y1, x2, y2, track_id = map(int, track)
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Verifica se o veículo está na free_area
            if free_area is not None and cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0:
                if track_id not in self.car_timers:
                    self.car_timers[track_id] = time.time()  # Inicia o timer para o veículo
                    self.car_moving_indices[track_id] = 0  # Inicializa o índice do veículo
                else:
                    elapsed_time = time.time() - self.car_timers[track_id]
                    if elapsed_time > self.T:
                        # Incrementa o índice com crescimento exponencial quadrático
                        normalized_time = (elapsed_time - self.T) / self.T
                        self.car_moving_indices[track_id] = min(1, normalized_time ** 2)
            else:
                # Remove o veículo do dicionário se ele sair da free_area
                if track_id in self.car_timers:
                    del self.car_timers[track_id]
                    del self.car_moving_indices[track_id]

            # Calcula o índice final para o veículo
            if track_id in self.car_moving_indices:
                car_index = self.car_moving_indices[track_id]
                free_slots_percentage = self.pr_info["Available"] / len([region for region in self.json if "points" in region])
                if car_index > 0:
                    # Calcula o tempo normalizado
                    normalized_time = (time.time() - self.car_timers[track_id] - self.T) * car_index * peso_car_moving / self.T
                    # Calcula o índice final com crescimento exponencial baseado na ocupação
                    if free_slots_percentage == 0: # estacionamento cheio
                        final_index = min(1, normalized_time ** (1 + full_park_index))
                    else:
                        final_index = min(1, (normalized_time ** (1 + free_slots_percentage)))
                else:
                    final_index = 0
                vehicle_indices.append((track_id, final_index))

        # Remove IDs de veículos que não estão mais na cena
        active_ids = [int(track[4]) for track in tracked_vehicles] if 'tracked_vehicles' in locals() else []
        self.car_timers = {k: v for k, v in self.car_timers.items() if k in active_ids}
        self.car_moving_indices = {k: v for k, v in self.car_moving_indices.items() if k in active_ids}

        # Desenha a tabela no canto superior direito
        start_x, start_y = im0.shape[1] - 300, 10  # Posição inicial da tabela
        row_height = 25
        cv2.rectangle(im0, (start_x, start_y), (im0.shape[1] - 10, start_y + (len(vehicle_indices) + 1) * row_height), (50, 50, 50), -1)
        cv2.putText(im0, "ID   Index", (start_x + 10, start_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        for i, (track_id, final_index) in enumerate(vehicle_indices):
            text = f"{track_id:<4} {final_index:.2f}"
            cv2.putText(im0, text, (start_x + 10, start_y + (i + 2) * row_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Count cars in the free_area
        for track in tracked_vehicles if 'tracked_vehicles' in locals() else []:
            x1, y1, x2, y2, _ = map(int, track)
            xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Check if the car is inside the free_area
            if free_area is not None:
                if cv2.pointPolygonTest(free_area, (xc, yc), False) >= 0:
                    cars_in_free_area += 1

        # Process parking regions (excluding free_area)
        for region in self.json:
            if "points" not in region:  # Skip regions without "points"
                continue

            # Convert points to a NumPy array with the correct dtype and reshape properly
            pts_array = np.array(region["points"], dtype=np.int32).reshape((-1, 1, 2))
            rg_occupied = False  # occupied region initialization
            
            # Verifica se algum veículo está na vaga
            for track in tracked_vehicles if 'tracked_vehicles' in locals() else []:
                x1, y1, x2, y2, _ = map(int, track)
                if self.is_in_parking_spot([x1, y1, x2, y2], region["points"]):
                    rg_occupied = True
                    break
                    
            fs, es = (fs + 1, es - 1) if rg_occupied else (fs, es)
            # Plotting regions
            cv2.polylines(im0, [pts_array], isClosed=True, color=self.occ if rg_occupied else self.arc, thickness=2)

        self.pr_info["Occupancy"], self.pr_info["Available"] = fs, len([region for region in self.json if "points" in region]) - fs

        # Display parking information
        cv2.putText(
            im0,
            f"Occupied: {self.pr_info['Occupancy']}, Available: {self.pr_info['Available']}",
            (10, 30),  # Position on the image
            cv2.FONT_HERSHEY_SIMPLEX,  # Font
            1,  # Font scale
            (104, 31, 17),  # Text color
            2,  # Thickness
            cv2.LINE_AA,  # Line type
        )

        # Display cars in free_area
        if free_area is not None:
            cv2.putText(
                im0,
                f"Cars in Free Area: {cars_in_free_area}",
                (10, 60),  # Position on the image
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Text color
                2,
                cv2.LINE_AA,
            )
            # Draw the free_area polygon
            cv2.polylines(im0, [free_area], isClosed=True, color=(0, 255, 0), thickness=2)

        self.display_output(im0)  # display output with base class function
        return im0  # return output image for more usage