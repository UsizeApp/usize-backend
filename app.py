import os
from time import sleep

import flask_wtf
import flask_wtf.file
import wtforms
import pandas as pd

from flask import (Flask, escape, jsonify, redirect, render_template, request, session, url_for)
from flask_migrate import Migrate
from flask_restful import Api, Resource, abort, reqparse
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from werkzeug.datastructures import FileStorage
from wtforms import IntegerField, PasswordField, StringField
from wtforms.validators import DataRequired

################################################
# Mensaje de inicio
################################################
print(
'''
===========================
Usize - API
===========================
''')

################################################
# Flask
################################################
DBG = 1

app = Flask(__name__)

DB_FILE = 'usize.sqlite'
DB_URI = 'sqlite:///%s' % DB_FILE

secret_key = b'_5#y2L"Fqqqqqa14Q8z\n\xec]/'

if os.path.isfile('SECRET_KEY'):
    with open('SECRET_KEY') as f:
        secret_key = f.read().strip()
        secret_key = bytes(secret_key, encoding='utf-8')

app.config.update(
    #TESTING=True,
    SECRET_KEY=secret_key,
    SQLALCHEMY_DATABASE_URI = DB_URI,
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    TEMPLATES_AUTO_RELOAD = True
)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

api = Api(app, catch_all_404s=True)

# Logger
LOG_ENABLED = 1


# Nuevo resource con GET por defecto y sistema de logging
class NewResource(Resource):
    LOG_TAG = "Default"
    bMostrarSolicitud = False

    def log(self, msg=""):
        if LOG_ENABLED: print(">>> [%s] %s" % (self.LOG_TAG, msg))

    def get(self):
        return {'message': 'GET %s' % self.__class__.__name__}
        
    def post(self):
        self.log('POST %s' % self.__class__.__name__)

        if self.bMostrarSolicitud:
            self.log(request)
            self.log(request.args)
            self.log(request.values)
            self.log(request.headers)
    

class v2Login(NewResource):
    LOG_TAG = "Login"

    def post(self):
        parser = reqparse.RequestParser()
        
        parser.add_argument('email', help="email missing", required=True)
        parser.add_argument('pwd', help="pwd missing", required=True)

        args = parser.parse_args()

        email = args['email']
        pwd = args['pwd']

        self.log("%s:%s" % (email, pwd))

        token = None

        u = Email.buscarEmail(email)
        if u is not None:
            if u.validPassword(pwd):
                self.log("%s autenticado" % u)
                token = makeToken(u)
            else:
                self.log("Password incorrecta")        
        else:
            self.log("Email no existe")

        return {'token': token}


class v2Medidas(NewResource):
    LOG_TAG = "Medidas"

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('token', help="token missing", required=True, location='headers')
        
        parser.add_argument('persona', help="persona", required=True, location='headers')
        
        args = parser.parse_args()
        token = args['token']
        persona = args['persona']

        self.log("persona %s" % (persona))

        p = Persona.get(persona)
        medidas = None

        if p is None:
            self.log("Persona no encontrada")
        else:
            medidas = p.getMedidas()

        return {'medidas': medidas}

class apiTallas(NewResource):
    LOG_TAG = "Tallas"

    #TODO mover como columnas a la Persona?

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('id', help="id_persona", required=True, location='headers')

        args = parser.parse_args()

        id_persona = args['id']

        tallas = None

        p = Persona.get(id_persona)
        if p is not None:
            self.log("Persona {}".format(p))
            tallas = getTallas(p)
        else:
            self.log("Persona no encontrada")

        return {'tallas': tallas}


class apiDatosEmail(NewResource):
    LOG_TAG = "apiDatosEmail"

    def post(self):
        super().post()

        parser = reqparse.RequestParser()

        parser.add_argument('token', help="token missing", required=True, location='headers')

        args = parser.parse_args()

        token = args['token']

        datosEmail = None

        email = decodeToken(token)
        if email is not None:
            self.log("Token {} con email {}".format(token, email))
            datosEmail = email.getDatos()
        else:
            self.log("Token {} sin email".format(token))

        return {'datosEmail': datosEmail}


class apiUpload(NewResource):
    """Docstring"""
    LOG_TAG = "apiUpload"

    def post(self):
        super().post()

        parser = reqparse.RequestParser()

        parser.add_argument('id', help="id_persona missing", required=True, location='headers')
        parser.add_argument("frontalphoto", help="photo missing", type=FileStorage, required=True, location='files')
        parser.add_argument("lateralphoto", help="photo missing", type=FileStorage, required=True, location='files')
        parser.add_argument("height", help="height missing", required=True)

        args = parser.parse_args()

        id_persona = args['id']
        frontal_photo = args['frontalphoto']
        lateral_photo = args['lateralphoto']
        height = args['height']

        self.log("id_persona: %s, frontal: %s, lateral: %s, height: %s" %
                 (id_persona, frontal_photo.filename, lateral_photo.filename, height))
        
        mensaje, medidas = tf.handle_photo(frontal_photo, lateral_photo, height)

        datosPersona = None
        
        if medidas is not None:
            try:
                p = Persona.get(id_persona)
                if p is not None:
                    self.log("Guardando medidas para Persona {}".format(p))
                    datosPersona = p.guardarMedidas(medidas)
                    self.log("Medidas guardadas")
                else:
                    self.log("Persona no encontrada")
            except:
                pass
        else:
            self.log("Sin medidas")
          
        return {"mensaje": mensaje, "datosPersona": datosPersona}


class apiRegister(NewResource):
    LOG_TAG = "Register"

    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('email', help="email missing", required=True)
        parser.add_argument('pwd', help="pwd missing", required=True)
        parser.add_argument('nombre')
        parser.add_argument('rut')
        parser.add_argument('gender')
        
        args = parser.parse_args()

        email = args['email']
        pwd = args['pwd']
        nombre = args['nombre']
        rut = args['rut']
        gender = args['gender']

        self.log("%s:%s:%s:%s:%s" % (email, pwd, nombre, rut, gender))

        respuesta = 'error'
        token = None

        u = Email.buscarEmail(email)        

        if u is not None:
            # El email ya existe
            self.log("Email ya existe")
            respuesta = 'ya_existe'
        else:
            # Lo creamos
            self.log("Email disponible")
            u, p = Email.addEmail(email=email, pwd=pwd, nombre=nombre, rut=rut, gender=gender)
            token = makeToken(u)
            respuesta = 'ok'

        return {"respuesta": respuesta, "token": token}


class apiValidarToken(NewResource):
    LOG_TAG = "apiValidarToken"

    def post(self):
        super().post()

        parser = reqparse.RequestParser()

        parser.add_argument('token', help='token', required=True, location='headers')

        args = parser.parse_args()

        token = args['token']
        respuesta = ''

        if token is not None:
            usuario = decodeToken(token)
            if usuario is not None:
                respuesta = 'valido'
                self.log("Token {} valido".format(token))
            else:
                respuesta = 'bad_token'
                self.log("Token {} sin usuario asociado".format(token))
        else:
            respuesta = 'bad_token'
            self.log("Token {} invalido".format(token))
                
        return {'respuesta': respuesta}


class apiDatosPersona(NewResource):
    LOG_TAG = "apiDatosPersona"

    def post(self):
        super().post()

        parser = reqparse.RequestParser()

        parser.add_argument('id', help='id', type=int, required=True, location='headers')

        args = parser.parse_args()

        id_persona = args['id']
        
        datosPersona = None
        
        p = Persona.get(id_persona)
        if p is not None:
            self.log('Persona: {}'.format(p))
            datosPersona = p.getDatos()
        else:
            self.log('Persona no encontrada')

        return {'datosPersona': datosPersona}


api.add_resource(v2Login, '/login')
api.add_resource(apiValidarToken, '/validarToken')

api.add_resource(apiDatosEmail, '/datosEmail')
api.add_resource(apiDatosPersona, '/datosPersona')

# Sube imagenes y genera las medidas
api.add_resource(apiUpload, '/upload')

# Registra un Email nuevo
api.add_resource(apiRegister, '/register')

# Despreciados
api.add_resource(v2Medidas, '/medidas')

api.add_resource(apiTallas, '/tallas')

######################
# Base de datos
# Uso de SQLite y SQLAlchemy
######################
from model import Email, Persona
from tf import tfManager

################################################
# TensorFlow
# Se instancia a la clase tfManager() para inicializar el modelo retinanet de deteccion de personas
# TF_ENABLED indica si es que se va a usar o no, para ganar tiempo en caso de debugging
# TODO Requests queue, since TF person_detector can't handle simultaneous requests
################################################
TF_ENABLED = 0
tf = tfManager(TF_ENABLED)
#from scripts.testpersona2 import a, b
#x = person_detector(photo)
#xx, yy = a()
#human = 'samples\\single2.jpg'

#from functools import wraps # TODO Implementar decoradores de auth

# deja abierto csv de tallas para consultas
df = pd.read_csv('tallas.csv')

# Tokens extremadamente basicos: solo la id del usuario
def makeToken(user):
    return str(user.id)


def decodeToken(token=None):
    if token is None:
        return None

    try:
        id = int(token)
        return Email.getEmailByID(id)
    except:
        return None


def getTallas(p):
    print("buscando tallas")
    left_arm = float(p.left_arm)
    right_arm = float(p.right_arm)

    left_leg = float(p.left_leg)
    right_leg = float(p.right_leg)

    waist = float(p.waist)
    hips = float(p.hips)
    chest = float(p.chest)
    bust = float(p.bust)

    medidas = p.getMedidas()

    gender = p.gender

    table = df.loc[df['genero'] == gender].values.tolist()

    print(table)
    return table

@app.errorhandler(404)
def page_not_found(e):
    return {"message": "404"}


# EOF
#############################################
