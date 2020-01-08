from imageai.Detection import ObjectDetection
import os

SPEED = "normal"  # normal, fast, fastest, flash


class MyDetector:
    detector = None
    custom_objects = None

    def __init__(self):
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()

        carpetaActual = os.getcwd()
        CARPETA = 'resources'
        MODELO = 'resnet50_coco_best_v2.0.1.h5'
        model_path = ''

        # Si la carpeta del modelo está aqui, la usamos
        if os.path.isdir(CARPETA):
            model_path = os.path.join(carpetaActual, CARPETA, MODELO)
        else:
            # Si no, buscamos en la carpeta superior
            CARPETA = '../' + CARPETA
            if os.path.isdir(CARPETA):
                model_path = os.path.join(carpetaActual, CARPETA, MODELO)
            else:
                # Error: la carpeta no se encuentra
                pass

        detector.setModelPath(model_path)
        detector.loadModel(detection_speed=SPEED)
        custom_objects = detector.CustomObjects(person=True)

        self.detector = detector
        self.custom_objects = custom_objects

    def person_detector(self, file):
        filename, file_ext = os.path.splitext(file)
        output = filename+"_detection"+file_ext

        detections = self.detector.detectCustomObjectsFromImage(
            custom_objects=self.custom_objects,
            input_image=file,
            output_image_path=output,
            minimum_percentage_probability=70)

        for eachObject in detections:
            object_type = eachObject["name"]
            #probability = eachObject["percentage_probability"]
            #print(object_type, " : ", probability, ":", eachObject["box_points"])

            if object_type == "person":
                return True
        return None


# Funcion antigua para detectar personas
# Genera los datos de TF en cada invocacion - bad!
def person_detector_old(file):
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    execution_path = os.getcwd()
    detector.setModelPath(os.path.join(
        execution_path, "resources/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel(detection_speed=SPEED)
    custom_objects = detector.CustomObjects(person=True)
    filename, file_ext = os.path.splitext(file)
    output = filename+"_detection"+file_ext
    detections = detector.detectCustomObjectsFromImage(
        custom_objects=custom_objects, input_image=file, output_image_path=output, minimum_percentage_probability=70)
    for eachObject in detections:
        object_type = eachObject["name"]
        probability = eachObject["percentage_probability"]
        print(object_type, " : ", probability, ":", eachObject["box_points"])
        if object_type == "person":
            return True
    return None
