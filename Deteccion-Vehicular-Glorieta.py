import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
class Config:
    VIDEO_PATH = "Video-Prueba.mp4"
    MODEL_PATH = "best.pt"
    OUTPUT_PATH = "Video-Salida.mp4"
    CSV_PATH = "recorridos.csv"
    
    # Parámetros de Trackeo
    UMBRAL_CONFIANZA = 0.6        # Si un objeto esta por debajo del umbral no lo detecta
    DISTANCIA_MAXIMA_MATCH = 70   # Pixeles maximos para considerar que es el mismo objeto
    MAX_FRAMES_PERDIDO = 100      # Cuantos frames aguanta el "fantasma" antes de morir
    ALPHA_SUAVIZADO = 0.2         # Para el filtro de velocidad
    OMEGA_MIN = 0.005
    VELOCIDAD_QUIETO = 0.01
    MARGEN_LIM_GLORIETA = 2
    
    # Geometría
    CENTRO_GLORIETA = (639, 361) # Centro en Pixeles
    
    # Poligonos que definen el contorno de la glorieta
    POLIG_AEREO = np.array([
        [384, 697], [446, 710], [665, 694], [728, 634], [763, 608],
        [841, 518], [802, 384], [771, 267], [680, 117], [545, 76],
        [441, 70], [248, 180], [230, 218], [182, 320], [295, 667]
    ], np.int32)

    POLIG_LADO = np.array([
        [776, 324], [729, 315], [558, 309], [503, 319], [462, 324],
        [381, 340], [352, 381], [325, 422], [335, 507], [488, 548],
        [636, 565], [930, 522], [947, 502], [973, 443], [843, 331]
    ], np.int32)

# ==========================================
# UTILIDADES GEOMÉTRICAS
# ==========================================
class GeoUtils:
    def __init__(self):
        # Toma el polígono en la cámara y el polígono aéreo y calcula de Homografia
        self.H, _ = cv2.findHomography(Config.POLIG_LADO, Config.POLIG_AEREO)
        self.H_inv = np.linalg.inv(self.H)

    def to_aereo(self, x, y):
        # Para saber si el coche está dentro o no
        # Para calcular la velocidad angular, solo en el mapa aéreo tiene sentido
        p = np.array([[[x, y]]], dtype='float32')
        res = cv2.perspectiveTransform(p, self.H)
        return res[0][0]

    def to_inclinado(self, x_a, y_a):
        # Para saber donde pintar el recuadro en el video original
        p = np.array([[[x_a, y_a]]], dtype='float32')
        res = cv2.perspectiveTransform(p, self.H_inv)
        return res[0][0]

    @staticmethod
    def punto_en_poligono(punto, poligono):
        # Saber si un carro está dentro o fuera de la glorieta
        return cv2.pointPolygonTest(poligono, punto, False) >= 0

geo = GeoUtils() # Se inicializa para no tener que recalcular la matriz H

# ==========================================
# DEFINICIÓN DE ZONAS (ENTRADAS Y SALIDAS)
# ==========================================
ZONAS = {
    "ENTRADAS": {
        "1": [[468, 327],[395, 343]], "2": [[325, 425],[342, 498]],
        "3": [[640, 563],[819, 533]], "4": [[946, 500],[966, 435]],
        "5": [[727, 317],[564, 313]]
    },
    "SALIDAS": {
        "1": [[553, 309],[510, 317]], "2": [[363, 344],[345, 377]],
        "3": [[341, 516],[483, 542]], "4": [[821, 529],[927, 514]],
        "5": [[840, 330],[779, 321]]
    }
}

def checar_cercania_zona(x, y, tipo):
    min_dist = float('inf')
    zona_detectada = None
    
    for nombre, coords in ZONAS[tipo].items():
        linea = np.array(coords, np.int32)
        dist = abs(cv2.pointPolygonTest(linea, (x, y), True))
        
        if tipo == "ENTRADAS":
            # ENTRADAS: Gana la más cercana sin importar distancia
            if dist < min_dist:
                min_dist = dist
                zona_detectada = nombre
        
        else:
            # === SALIDAS CON CORRECCIÓN DE PERSPECTIVA ===
            
            # Calculamos el umbral según la altura (y) del carro.
            # La fórmula es: El 5% de su altura en pantalla.
            umbral_dinamico = int(y * 0.05)
            
            # Para que no sea ni ridículamente pequeño ni gigante
            umbral_dinamico = max(15, min(30, umbral_dinamico))

            if dist < umbral_dinamico and dist < min_dist:
                min_dist = dist
                zona_detectada = nombre
    
    return zona_detectada

# ==========================================
# CLASE VEHICULO
# ==========================================
class Vehiculo:
    #//// La funcion que inicializa cada objeto una vez
    def __init__(self, id_obj, deteccion):
        self.id = id_obj
        self.edad = 0
        self.actualizar_datos(deteccion)
        
        self.velocidad = np.array([0.0, 0.0]) # vx, vy
        self.omega = 0.0
        self.frames_perdido = 0
        self.trayectoria = [] # Historial de puntos
        
        self.entrada = checar_cercania_zona(self.cx, self.cy, "ENTRADAS") # Saber en que entrada "nacio"
        self.salida = None
        self.contado = False # Para no contar el mismo coche dos veces
        
        
    #//// Actualizar datos en cada iteracion
    def actualizar_datos(self, deteccion):
        self.cx, self.cy = deteccion['centro']  # Actualiza mi posicion actual
        self.bbox = deteccion['bbox']
        self.color = deteccion['color']
        self.clase = deteccion['clase']
        self.conf = deteccion['conf']
        self.cx_aereo, self.cy_aereo = geo.to_aereo(self.cx, self.cy)
        self.edad += 1

    #//// Predecir el movimiento de los fantasmas
    def predecir_posicion(self):
        # Si mi velocidad es baja, asumo que esta quieto
        if np.linalg.norm(self.velocidad) < 0.05:
            return self.cx, self.cy, self.cx_aereo, self.cy_aereo

        dx = self.cx_aereo - Config.CENTRO_GLORIETA[0]
        dy = Config.CENTRO_GLORIETA[1] - self.cy_aereo
        radio = (dx**2 + dy**2)**0.5
        
        # Umbral para decidir si es movimiento lineal o rotacional
        if abs(self.omega) < Config.OMEGA_MIN: 
            # Movimiento Lineal
            pred_x_a = self.cx_aereo + self.velocidad[0]
            pred_y_a = self.cy_aereo + self.velocidad[1]
        else: 
            # Movimiento Rotacional 
            theta = np.arctan2(-dy, dx)
            theta_nuevo = theta + self.omega
            pred_x_a = Config.CENTRO_GLORIETA[0] + radio * np.cos(theta_nuevo)
            pred_y_a = Config.CENTRO_GLORIETA[1] + radio * np.sin(theta_nuevo) 
            
        # Convertir prediccion aérea de vuelta a coordenadas de imagen
        px, py = geo.to_inclinado(pred_x_a, pred_y_a)
        return px, py, pred_x_a, pred_y_a

    #//// Se ejecuta solo cuando logramos reencontrar al vehículo en el frame actual
    def actualizar_pos_real(self, nueva_deteccion):
        # 1. Calcular velocidades antes de actualizar posición
        dt = 1 # Asumimos 1 frame de diferencia
        
        # Posiciones actuales (aereas)
        new_cx, new_cy = nueva_deteccion['centro']
        new_cx_a, new_cy_a = geo.to_aereo(new_cx, new_cy)
        
        # Calculo vector velocidad (con suavizado)
        vx_raw = new_cx_a - self.cx_aereo
        vy_raw = new_cy_a - self.cy_aereo
        
        self.velocidad = (Config.ALPHA_SUAVIZADO * np.array([vx_raw, vy_raw]) + 
                          (1 - Config.ALPHA_SUAVIZADO) * self.velocidad)
        
        # Calculo velocidad angular (Omega)
        dx = self.cx_aereo - Config.CENTRO_GLORIETA[0]
        dy = Config.CENTRO_GLORIETA[1] - self.cy_aereo
        r_sq = dx**2 + dy**2 + 1e-6
        omega_inst = ((self.velocidad[0] * dy) + (self.velocidad[1] * dx)) / r_sq
        self.omega = 0.7 * self.omega + 0.3 * omega_inst # Suavizado

        # 2. Actualizar estado
        self.actualizar_datos(nueva_deteccion)
        self.frames_perdido = 0
        self.trayectoria.append((self.cx, self.cy))
        
        # 3. Checar entrada si aun no tiene
        if self.entrada is None:
            self.entrada = checar_cercania_zona(self.cx, self.cy, "ENTRADAS")

    
    #//// Cuando un vehiculo no se ve, esta funcion lo mantiene vivo
    def marcar_perdido(self):
        # Usar la prediccion para "mover" al fantasma
        pred_x, pred_y, pred_x_a, pred_y_a = self.predecir_posicion()
        
        # Actualizamos sus coordenadas a las predichas
        bbox_w, bbox_h = self.bbox[2], self.bbox[3]
        sim_bbox = (int(pred_x - bbox_w/2), int(pred_y - bbox_h/2), bbox_w, bbox_h)
        
        simulacion = {
            'centro': (pred_x, pred_y),
            'bbox': sim_bbox,
            'color': self.color,
            'clase': self.clase,
            'conf': 0.0 # Es un fantasma
        }
        
        # Actualizamos datos internos 
        self.actualizar_datos(simulacion)
        self.cx_aereo = pred_x_a
        self.cy_aereo = pred_y_a
        self.frames_perdido += 1
        self.edad += 1
        
        # Checar si salió mientras era fantasma
        if self.salida is None:
            self.salida = checar_cercania_zona(pred_x, pred_y, "SALIDAS")

# ==========================================
# CLASE TRACKER (GESTOR DE TODOS LOS COCHES)
# ==========================================
class Tracker:
    #//// Inicializar el objeto de Tracker
    def __init__(self):
        self.vehiculos = []
        self.next_id = 0
        self.conteo_recorridos = {} # {(entrada, salida): count}

    #//// Asociacion de Datos
    def rastrear(self, detecciones):
        # 1. Predecir ubicaciones de vehículos existentes
        predicciones = []
        for v in self.vehiculos:
            px, py, _, _ = v.predecir_posicion()
            predicciones.append((px, py))

        # 2. Matriz de Costos (Distancia entre Predicciones y Nuevas Detecciones)
        if len(self.vehiculos) > 0 and len(detecciones) > 0:
            cost_matrix = np.zeros((len(self.vehiculos), len(detecciones)))
            
            for i, (px, py) in enumerate(predicciones):
                for j, det in enumerate(detecciones):
                    dx, dy = det['centro']
                    dist = ((px - dx)**2 + (py - dy)**2)**0.5
                    
                    cost_matrix[i, j] = dist
            
        # 3. Algoritmo de distancias entre vehiculos
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
        else:
            row_inds, col_inds = [], []

        assigned_tracks = set()
        assigned_dets = set()

        # 4. Actualizar matches válidos
        for r, c in zip(row_inds, col_inds):
            if cost_matrix[r, c] < Config.DISTANCIA_MAXIMA_MATCH:
                self.vehiculos[r].actualizar_pos_real(detecciones[c]) # Agarrar el coche viejo e inyectar la info                                     
                                                                      # a la nueva deteccion
                # lista de los Coches Viejos que SÍ encontraron pareja
                assigned_tracks.add(r)                 
                # lista de las Detecciones Nuevas que YA se usaron               
                assigned_dets.add(c)
        
        # 5. Manejar Vehículos NO detectados (Fantasmas)
        for i, vehiculo in enumerate(self.vehiculos):
            if i not in assigned_tracks:
                vehiculo.marcar_perdido()

        # 6. Crear nuevos vehículos 
        for j, det in enumerate(detecciones):
            if j not in assigned_dets:
                # Solo crear si está cerca de una entrada o borde 
                nuevo_v = Vehiculo(self.next_id, det) # Llama a init de el objeto Vehiculo, crea un expediente nuevo 
                self.vehiculos.append(nuevo_v)        # Lo anota en la lista 
                self.next_id += 1
        
        # 7. Limpieza y Conteo
        self.eliminar_clones_encimados()
        self.limpiar_y_contar()

    def limpiar_y_contar(self):
        activos = []
        
        for v in self.vehiculos:
            # 1. ESTADO DEL CARRO
            dist_borde = cv2.pointPolygonTest(Config.POLIG_AEREO, (v.cx_aereo, v.cy_aereo), True)
            esta_en_el_borde = dist_borde < Config.MARGEN_LIM_GLORIETA
            
            # CONDICIONES DE MUERTE
            es_viejo_y_esta_saliendo = (v.edad > 15) and esta_en_el_borde
            timeout_fantasma = v.frames_perdido >= Config.MAX_FRAMES_PERDIDO
            
            # ====================================================================
            # Si el carro se está saliendo AHORITA MISMO y no tiene salida asignarle una
            if es_viejo_y_esta_saliendo and v.salida is None:
                # Buscamos la salida más cercana a la fuerza
                min_dist = float('inf')
                mejor_salida = None
                
                for nombre, coords in ZONAS["SALIDAS"].items():
                    linea = np.array(coords, np.int32)
                    d = abs(cv2.pointPolygonTest(linea, (v.cx, v.cy), True))
                    if d < min_dist:
                        min_dist = d
                        mejor_salida = nombre
                
                # Le asignamos la salida más cercana 
                if min_dist < 20:
                    v.salida = mejor_salida
                    
            # ====================================================================
            # Guardar y contar
            ya_termino_viaje = False
            
            if v.salida is not None and v.entrada is not None:
                if not v.contado:
                    # REGISTRAR EN EL DICCIONARIO
                    key = (v.entrada, v.salida)
                    self.conteo_recorridos[key] = self.conteo_recorridos.get(key, 0) + 1
                    v.contado = True
                    print(f" FINALIZADO ID {v.id}: {v.entrada} -> {v.salida}")
                    
                    # GUARDAR CSV
                    try:
                        df_temp = pd.DataFrame([(e, s, c) for (e, s), c in self.conteo_recorridos.items()], columns=["Entrada", "Salida", "Conteo"])
                        df_temp.to_csv(Config.CSV_PATH, index=False)
                    except: pass
                
                ya_termino_viaje = True

            # ====================================================================
            # Filtro de vida, si no se cumple se borra
            if not ya_termino_viaje and not timeout_fantasma and not es_viejo_y_esta_saliendo:
                activos.append(v)
        
        self.vehiculos = activos

    def eliminar_clones_encimados(self):
        ids_a_eliminar = set()
        
        # Comparamos todos contra todos
        for i in range(len(self.vehiculos)):
            for j in range(i + 1, len(self.vehiculos)):
                v1 = self.vehiculos[i]
                v2 = self.vehiculos[j]
                
                # Si alguno ya está marcado para morir, lo ignoramos
                if v1.id in ids_a_eliminar or v2.id in ids_a_eliminar:
                    continue

                # Calculamos distancia entre ellos 
                dist = ((v1.cx - v2.cx)**2 + (v1.cy - v2.cy)**2)**0.5
                
                # SI ESTÁN A MENOS DE 5 PIXELES Están encimados
                if dist < 5:
                    # DECIDIR A QUIÉN MATAR:
                    if v1.edad < v2.edad:
                        ids_a_eliminar.add(v1.id)
                    else:
                        ids_a_eliminar.add(v2.id)

        # Reconstruir la lista quitando los clones
        self.vehiculos = [v for v in self.vehiculos if v.id not in ids_a_eliminar]


# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(Config.VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_out = cv2.VideoWriter(Config.OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
modelo = YOLO(Config.MODEL_PATH)
tracker = Tracker() # Ejecuta el init de la clase tracker ( Donde crea la lista de vehículos)

print("Iniciando procesamiento")
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    #/////////// 1. DETECCIÓN YOLO
    # YOLO devuelve una lista de todo lo que vio
    results = modelo(frame, verbose=False, conf=Config.UMBRAL_CONFIANZA, iou=0.4, agnostic_nms=True)[0] 
    # Aqui entran solo los coches válidos de la foto
    detecciones_frame = [] 

    for box in results.boxes: 
        # Obtener coordenadas en la imagen 
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
    
        # Filtro de glorieta para ver si está fuera o dentro
        cx_aereo, cy_aereo = geo.to_aereo(cx, cy)
        
        # Verificar si ese punto aéreo está dentro del Polígono Aéreo
        # cv2.pointPolygonTest devuelve > 0 si está dentro, < 0 si está fuera
        esta_dentro = cv2.pointPolygonTest(Config.POLIG_AEREO, (cx_aereo, cy_aereo), False)
        
        # Si está fuera, saltamos al siguiente objeto y lo ignoramos
        if esta_dentro < 0: 
            continue 
            
        # Extraer color promedio 
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        if roi.size == 0: continue # Evitar error si la caja está mal
        color = np.mean(roi, axis=(0, 1))

        # Obtenida toda la infromación se guarda
        detecciones_frame.append({
            'centro': (cx, cy),
            'bbox': (x1, y1, w, h),
            'color': color,
            'clase': int(box.cls[0]),
            'conf': float(box.conf[0])
        })

    #/////////// 2. ACTUALIZAR TRACKER (Solo con los que pasaron el filtro)
    tracker.rastrear(detecciones_frame)

    #/////////// 3. DIBUJAR EL POLIGONO DE LA GLORIETA
    cv2.polylines(frame, [Config.POLIG_LADO], True, (0, 255, 255), 2)
    
    # Dibujar entradas
    for tipo, lista_zonas in ZONAS.items():
        for nombre, coords in lista_zonas.items():
            # Las coordenadas vienen como lista de listas [[x1,y1], [x2,y2]]
            p1 = tuple(coords[0])
            p2 = tuple(coords[1])
            
            # Calcular el punto medio para poner el texto centrado
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            
            # Configurar colores y texto
            if tipo == "ENTRADAS":
                color = (255, 255, 0) # Cyan para Entradas
                texto = f"E{nombre}"  # Ejem: E1
            else:
                color = (255, 0, 255) # Magenta para Salidas
                texto = f"S{nombre}"  # Ejem: S1
            
            # 1. Dibujar la línea invisible (para que veas el "cable" que pisa el coche)
            cv2.line(frame, p1, p2, color, 2)
            
            # 2. Poner el texto
            cv2.putText(frame, texto, (mid_x - 10, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Dibujar casillas de los vehiculos
    for v in tracker.vehiculos:
        x, y, w, h = v.bbox
        x, y = int(x), int(y)
        cx, cy = int(v.cx), int(v.cy)
        
        # Color: Verde (Trackeado), Rojo (Fantasma/Perdido)
        color_rect = (0, 255, 0) if v.frames_perdido == 0 else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), color_rect, 2)
        cv2.putText(frame, f"ID:{v.id} {v.entrada}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rect, 2)
        
        # Dibujar centro
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    # Escribir frame
    video_out.write(frame)

# ==========================================
# EXPORTAR CSV
# ==========================================
df = pd.DataFrame(
    [(e, s, c) for (e, s), c in tracker.conteo_recorridos.items()],
    columns=["Entrada", "Salida", "Conteo"]
)
df.to_csv(Config.CSV_PATH, index=False)
print("Proceso terminado. CSV exportado.")

# Tiempo
end = time.time()    
print("Tiempo transcurrido:", end - start, "segundos") 

cap.release()
video_out.release()