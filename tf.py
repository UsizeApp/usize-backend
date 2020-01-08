import time
import os
from datetime import datetime as dt

from utils import printd, enablePrint

DBG = 1
enablePrint(DBG)

from scripts.testpersona import MyDetector
from scripts.get_measures import get_measurements

class tfManager:
    PHOTO_FOLDER = '%s%s' % (os.getcwd(), '\\input')
    TF_ENABLED = 0

    bIgnorarHumano = True

    def __init__(self, TF_ENABLED=False):
        self.TF_ENABLED = TF_ENABLED

        if not os.path.exists(self.PHOTO_FOLDER):
            os.makedirs(self.PHOTO_FOLDER)

        if TF_ENABLED:
            print(self.PHOTO_FOLDER)
            # Cargamos el modelo detector de personas
            t1 = time.time()
            self.mDetector = MyDetector()
            t2 = time.time()
            printd('TF iniciado en %.1f [s]' % (t2-t1))
        else:
            printd('TF sin iniciarse')
            pass

    def detectarHumano(self, full_frontal_photo_path):
        if self.bIgnorarHumano:
            return True

        elif self.TF_ENABLED:
            return self.mDetector.person_detector(full_frontal_photo_path)

        else:
            return False


    def handle_photo(self, frontal_photo, lateral_photo, height):
        medidas = None
        mensaje = None
        
        if height is None:
            height = 174
        elif isinstance(height, str):
            try:
                height = int(height)
            except:
                height = 174

        szNow = dt.now().strftime("%Y-%m-%d_%H.%M.%S")

        full_frontal_photo_path = ''
        full_lateral_photo_path = ''    
        
        bUsarFotosLocales = True
        if bUsarFotosLocales:
            # Usar fotos locales sin guardar
            full_frontal_photo_path = os.path.abspath('samples\\s1_frontal.jpg')
            full_lateral_photo_path = os.path.abspath('samples\\s1_lateral.jpg')
        else:
            # Preparar las rutas de las fotos y guardarlas
            frontal_photo_path = "%s\\%s_frontal.jpg" % (self.PHOTO_FOLDER, szNow)
            lateral_photo_path = "%s\\%s_lateral.jpg" % (self.PHOTO_FOLDER, szNow)
        
            frontal_photo.save(frontal_photo_path)
            lateral_photo.save(lateral_photo_path)

            full_frontal_photo_path = os.path.abspath(frontal_photo_path)
            full_lateral_photo_path = os.path.abspath(lateral_photo_path)

        printd('full_frontal_photo_path:', full_frontal_photo_path)
        printd('full_lateral_photo_path:', full_lateral_photo_path)

        EXTRA_TRIES = 3
        TRIES = 1 or EXTRA_TRIES

        # Si pPersona >= 0.7...
        if self.detectarHumano(full_frontal_photo_path):
            printd('yes_human')

            # Intentamos calcular las medidas TRIES veces
            for i in range(1, TRIES+1):
                printd("Intento %d" % i)

                try:
                    mensaje = 'Éxito al obtener las medidas'
                    medidas = get_measurements(full_frontal_photo_path, full_lateral_photo_path, height)
                    break
                # Si la medicion falla...
                except Exception as e:
                    mensaje = "No pudimos tomar las medidas. ¡Intenta otra vez!"
                    printd(e)
        # Si pPersona < 0.7
        else:
            mensaje = "No pudimos detectar a una persona."
        
        printd('mensaje:', mensaje)
        printd('medidas:', medidas)

        return mensaje, medidas
