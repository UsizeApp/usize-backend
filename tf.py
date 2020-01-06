import time
import os
from datetime import datetime as dt

DBG = 1

class tfManager:
    PHOTO_FOLDER = '%s%s' % (os.getcwd(), '\\input')
    TF_ENABLED = 0

    def __init__(self, TF_ENABLED=False):
        self.TF_ENABLED = TF_ENABLED

        if not os.path.exists(self.PHOTO_FOLDER):
            os.makedirs(self.PHOTO_FOLDER)

        if TF_ENABLED:
            print(self.PHOTO_FOLDER)
            # Si se debe iniciar TF, hacemos los imports
            from scripts.testpersona import MyDetector
            from scripts.OpenPoseImageV2 import open_pose_image
            from scripts.get_measures import get_measurements
            self.open_pose_image = open_pose_image

            # Y cargamos el modelo detector de personas
            t1 = time.time()
            self.mDetector = MyDetector()
            t2 = time.time()
            if DBG: print('TF iniciado en %.1f [s]' % (t2-t1))
        else:
            if DBG: print('TF sin iniciarse')
            pass


    def handle_photo(self, frontal_photo, lateral_photo, height):
        medidas = None
        mensaje = None
        
        try:
            height = int(height)
        except:
            height = 174                   
        szNow = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
        frontal_photo_path = "%s\\frontal_photo_%s.jpg" % (self.PHOTO_FOLDER, szNow)
        lateral_photo_path = "%s\\lateral_photo_%s.jpg" % (self.PHOTO_FOLDER, szNow)
        frontal_photo.save(frontal_photo_path)
        lateral_photo.save(lateral_photo_path)
        full_frontal_photo_path = os.path.abspath(frontal_photo_path)
        print(">> Saved: %s" % full_frontal_photo_path)
        full_lateral_photo_path = os.path.abspath(lateral_photo_path)
        print(">> Saved: %s" % full_lateral_photo_path)

        if not self.TF_ENABLED:
            return medidas, "TensorFlow desactivado"

        EXTRA_TRIES = 3
        TRIES = 1 or EXTRA_TRIES

        # Si pPersona >= 0.7...
        # if person_detector(full_frontal_path):
        bIgnorarHumano = False
        if bIgnorarHumano or self.mDetector.person_detector(full_frontal_photo_path):
            if DBG: print('yes_human')
            # Intentamos calcular las medidas TRIES veces
            for i in range(TRIES):
                try:
                    #medidas = self.open_pose_image(full_frontal_photo_path ,height)
                    medidas = self.get_measurements(full_frontal_photo_path, full_lateral_photo_path ,height)
                    if DBG: print(medidas)

                    mensaje = 'success',
                    break
                # Si la medicion falla...
                except Exception as e:
                    mensaje = "No pudimos tomar las medidas. ¡Intenta otra vez!"

                    if (DBG):
                        print(e)
                        mensaje = "%s %s" % (mensaje, str(e))
        # Si pPersona < 0.7
        else:
            mensaje = "No pudimos detectar a una persona."
            if DBG:
                print('no_human')
                mensaje = "%s %s" % (mensaje, 'no_human')
        
        return medidas, mensaje
