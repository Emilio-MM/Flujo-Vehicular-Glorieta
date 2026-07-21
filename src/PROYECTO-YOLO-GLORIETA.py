import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO   

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
class Config:
    VIDEO_PATH = "VIDEO1.mp4" # video de entrada 
    OUTPUT_PATH = "VIDEO_SALIDA1.mp4" # video de salida
    CSV_PATH = "recorridos.csv" # archivo de salida
    
    # Cargar Modelo y forzar CUDA
    MODEL = YOLO("C:/Users/choco/Downloads/best2.pt").to('cuda')
    
    UMBRAL_CONFIANZA = 0.30
    RESOLUCION_INFERENCIA = 1280  
    EDAD_MINIMA = 5      
    
    # Poligonos de la glorieta principal
    POLIG_AEREO = np.array([[824, 239], [798, 193], [765, 154], [684, 85], [633, 48], [603, 19], [515, 38], [351, 74], [230, 146], [141, 437], [95, 495], [106, 551], [181, 588], [208, 657], [282, 755], [376, 797], [443, 816], [529, 821],[655, 827], [737, 783], [754, 694], [801, 616], [829, 515], [850, 353],
    ], np.int32)

    POLIG_LADO = np.array([[640, 225], [589, 242], [535, 264], [420, 321], [343, 367], [277, 387], [261, 472], [218, 675], [316, 912], [1222, 941], [1398, 1019], [1527, 936], [1534, 733], [1606, 636], [1626, 495], [1562, 389], [1500, 333], [1419, 278], [1353, 222], [1234, 186], [1142, 193], [1037, 178], [931, 187], [761, 200],
    ], np.int32)
    
# ==========================================
# UTILIDADES GEOMÉTRICAS
# ==========================================
class GeoUtils:
    def __init__(self): # matriz de transformación de coordenadas
        self.H, _ = cv2.findHomography(Config.POLIG_LADO, Config.POLIG_AEREO)

    def to_aereo(self, x, y): # transformación de coordenadas aereo
        p = np.array([[[x, y]]], dtype='float32')
        res = cv2.perspectiveTransform(p, self.H)
        return res[0][0]

geo = GeoUtils()

ZONAS = {
    "ENTRADAS": {
        "1_normal": [[535, 264], [420, 321], [451, 284], [444, 262], [427, 249], [422, 268], [414, 291], [397, 314], [375, 337], [343, 367], [277, 387], [139, 433], [156, 399], [205, 377], [243, 357], [287, 328], [320, 301], [343, 272], [353, 238], [330, 208], [303, 187], [284, 173], [306, 162], [339, 153], [376, 170], [417, 189], [456, 208], [496, 228],], 
        "1_desnivel": [[638, 292], [664, 313], [717, 341], [781, 363], [870, 385], [965, 395], [1073, 403], [1159, 396], [1231, 396], [1293, 392], [1296, 418], [1232, 426], [1158, 431], [1071, 435], [958, 428], [857, 413], [765, 389], [686, 363], [634, 337], [607, 308],], 
        "2": [[316, 912], [218, 675], [0, 690], [0, 918], [1222, 1016], [1398, 1019], [1222, 941],],
        "3": [[1606, 636], [1626, 495], [1749, 487], [1786, 515], [1825, 549], [1865, 577], [1917, 603], [1914, 846],], 
        "4_normal": [[1419, 278], [1353, 222], [1418, 235], [1497, 252], [1568, 267], [1634, 269], [1704, 271], [1782, 272], [1877, 272], [1917, 265], [1916, 304], [1862, 305], [1803, 311], [1720, 313], [1618, 311], [1516, 300],],
        "4_desnivel": [[717, 341], [781, 363], [870, 385], [965, 395], [1073, 403], [1159, 396], [1231, 396], [1293, 392], [1393, 383], [1372, 350], [1288, 359], [1227, 359], [1156, 359], [1080, 357], [975, 354], [881, 346], [811, 328], [752, 311], ],
        "5": [[845, 189], [930, 189], [981, 172], [1038, 149], [1081, 131], [1130, 114], [1170, 98], [1104, 59], [1058, 62], [994, 82], [949, 94], [883, 110], [807, 133], [749, 141], [699, 149], [653, 151], [599, 147], [559, 143], [532, 141], [520, 154], [552, 163], [605, 172], [661, 173], [726, 170], [783, 160]]
    },
    "SALIDAS": {
        "1_normal": [[622, 226], [701, 203], [661, 195], [612, 182], [561, 166], [522, 153], [540, 141], [502, 131], [466, 120], [431, 108], [394, 97], [374, 104], [352, 118], [392, 137], [445, 157], [496, 176], [556, 203],],
        "1_desnivel": [[638, 292], [664, 313], [717, 341], [752, 311], [710, 297], [666, 281],],
        "2": [[277, 387], [139, 433], [159, 390], [0, 448], [1, 582], [261, 472]],
        "3": [[1527, 936], [1534, 733], [1658, 764], [1776, 809], [1916, 877], [1916, 1033], [1398, 1019],], 
        "4_normal": [[1626, 495], [1562, 389], [1603, 367], [1677, 353], [1743, 351], [1834, 353], [1916, 353], [1914, 415], [1870, 415], [1814, 415], [1746, 425], [1724, 448], [1749, 487],],
        "4_desnivel": [[1293, 392], [1296, 418], [1398, 402], [1393, 383], ],
        "5": [[1353, 222], [1234, 186], [1142, 193], [1123, 164], [1124, 131], [1142, 114], [1166, 105], [1221, 118], [1265, 137], [1296, 150], [1288, 166], [1286, 185], [1296, 198], [1320, 213],]
    }
}

def obtener_entrada(x, y): # obtener la entrada del vehículo
    for nombre, coords in ZONAS["ENTRADAS"].items(): # recorrer las entradas
        poligono = np.array(coords, np.int32) # polígono de la entrada
        if cv2.pointPolygonTest(poligono, (x, y), False) >= 0:
            return nombre.split('_')[0]
    return None

def obtener_salida(x, y): # obtener la salida del vehículo
    for nombre, coords in ZONAS["SALIDAS"].items():
        poligono = np.array(coords, np.int32) # polígono de la salida
        if cv2.pointPolygonTest(poligono, (x, y), False) >= 0:
            return nombre.split('_')[0] 
    return None

# ==========================================
# GESTOR DE ESTADOS
# ==========================================
class GestorTrafico:
    def __init__(self): # diccionario de vehículos y sus estados
        self.memoria_vehiculos = {} 
        self.conteo_recorridos = {} # diccionario de recorridos

    def procesar_vehiculo(self, track_id, cx, cy, esta_dentro): 
        # Nace sin asignar entrada
        if track_id not in self.memoria_vehiculos:
            self.memoria_vehiculos[track_id] = { # estado inicial del vehículo
                'entrada': None, 
                'salida': None, 
                'contado': False, 
                'edad': 0
            }

        # Envejecer el vehículo
        datos = self.memoria_vehiculos[track_id]
        datos['edad'] += 1

        # LÓGICA DE ENTRADA (Cruce por polígono)
        # Si tienes None (naciste afuera) y aún no te han contado
        if datos['entrada'] is None and not datos['contado']:
            entrada_detectada = obtener_entrada(cx, cy)
            if entrada_detectada:
                datos['entrada'] = entrada_detectada

        # LÓGICA DE SALIDA
        if datos['entrada'] and datos['entrada'] != "IGNORAR" and not datos['contado'] and datos['edad'] >= Config.EDAD_MINIMA:
            salida = obtener_salida(cx, cy)
            
            # Si tocó el polígono de salida
            if salida:
                datos['salida'] = salida
                datos['contado'] = True
                
                key = (datos['entrada'], salida) # clave del recorrido
                self.conteo_recorridos[key] = self.conteo_recorridos.get(key, 0) + 1 # contador de recorridos
                print(f" FINALIZADO ID {track_id}: E{datos['entrada']} -> S{salida} (Vivió {datos['edad']} frames)") # mensaje de finalización
    
        
    def obtener_info(self, track_id):
        return self.memoria_vehiculos.get(track_id, None) # información del vehículo


# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(Config.VIDEO_PATH)
width, height = int(cap.get(3)), int(cap.get(4)) # ancho y alto del video
fps = cap.get(cv2.CAP_PROP_FPS) # frames por segundo

video_out = cv2.VideoWriter(Config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
gestor = GestorTrafico()
contador_frames = 0

print(f"Procesando en GPU: {Config.MODEL.device}")
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Tracking simple
    results = Config.MODEL.track(
        frame, 
        persist=True, 
        tracker="custom_tracker.yaml",     
        imgsz=Config.RESOLUCION_INFERENCIA, 
        conf=Config.UMBRAL_CONFIANZA,
        iou=0.5,
        verbose=False
    )[0]

    # Extracción de datos para cada bounding box
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().int().numpy()
        clases = results.boxes.cls.cpu().int().numpy()

        for box, track_id, cls in zip(boxes, track_ids, clases):
            
            if cls == 0:
                continue
            
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # centro del bounding box
            
            # Comprobar si está en la glorieta principal
            cx_a, cy_a = geo.to_aereo(cx, cy)
            esta_dentro = cv2.pointPolygonTest(Config.POLIG_AEREO, (cx_a, cy_a), False) >= 0

            # El gestor ahora evalúa la posición actual para saber si ya debe calcular la entrada
            gestor.procesar_vehiculo(track_id, cx, cy, esta_dentro)
            
            info = gestor.obtener_info(track_id)

            # === FILTRO VISUAL ===
            # Solo dibujar si está ADENTRO y NO HA SIDO CONTADO AÚN
            if (esta_dentro or info['entrada']) and not info['contado']:
                color = (0, 255, 0)
                texto_entrada = f" E{info['entrada']}" if info['entrada'] else ""

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                cv2.putText(frame, f"ID:{track_id}{texto_entrada}", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # OVERLAY DE GLORIETA PRINCIPAL
    cv2.polylines(frame, [Config.POLIG_LADO], True, (0, 255, 255), 2)
    
    # DIBUJAR ETRADAS Y SALIDAS
    for tipo, lista_zonas in ZONAS.items():
        for nombre, coords in lista_zonas.items():
            pts = np.array(coords, np.int32)
            
            # Calcular centro del polígono para el texto
            M = cv2.moments(pts)
            if M["m00"] != 0:
                mid_x = int(M["m10"] / M["m00"])
                mid_y = int(M["m01"] / M["m00"])
            else:
                mid_x, mid_y = tuple(coords[0])
            
            if tipo == "ENTRADAS":
                color = (255, 255, 0) 
                texto = f"E{nombre.split('_')[0]}"
            else:
                color = (255, 0, 255) # M'_')[0]}"
                
            # Dibujar caja y texto
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, texto, (mid_x - 10, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    video_out.write(frame)


# Exportación final
df = pd.DataFrame([(e, s, c) for (e, s), c in gestor.conteo_recorridos.items()], 
                  columns=["Entrada", "Salida", "Conteo"])
df.to_csv(Config.CSV_PATH, index=False)

end = time.time()    
print("Tiempo transcurrido:", end - start, "segundos") 

cap.release()
video_out.release()
print("Proceso finalizado")