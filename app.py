import os
from time import sleep

import flask_wtf
import flask_wtf.file
import wtforms

from flask import (Flask, escape, jsonify, redirect, render_template, request, session, url_for)
from flask_migrate import Migrate
from flask_restful import Api, Resource, abort, reqparse
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from werkzeug.datastructures import FileStorage
from wtforms import IntegerField, PasswordField, StringField
from wtforms.validators import DataRequired

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

	def log(self, msg=""):
		if LOG_ENABLED: print(">>> [%s] %s" % (self.LOG_TAG, msg))

	def get(self):
		return {'message': 'GET %s' % self.__class__.__name__}
	

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

		u = Usuario.getUsuarioByEmail(email)
		if u is None:
			self.log("Email no existe")
			return abort(401, message="Bad login")
		if not u.validPassword(pwd):
			self.log("Password incorrecta")
			return abort(401, message="Bad login")
		self.log("%s autenticado" % u)		
		return {"status": "ok", "token": str(u.id)}


class v2Medidas(NewResource):
	LOG_TAG = "Medidas"

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('token', help="token missing", required=True, location='headers')
		args = parser.parse_args()
		token = args['token']

		self.log("Token %s" % (token))

		u = decodeToken(token)
		if u is None:
			self.log("Usuario no encontrado")
			return abort(401, message="Bad token")
		self.log("%s autorizado" % u)
		return {"medidas": {
					'right_arm': u.right_arm,
					'left_arm': u.left_arm,
					'right_leg': u.right_leg,
					'left_leg': u.left_leg,
					'waist': u.waist,
					'hip': u.hip,
					'chest': u.chest,
					'bust': u.bust,}
		}


class v2Perfil(NewResource):
	LOG_TAG = "Perfil"

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('token', help="token missing", required=True, location='headers')
		args = parser.parse_args()
		token = args['token']

		self.log("Token %s" % (token))

		u = decodeToken(token)
		if u is None:
			self.log("Usuario no encontrado")
			return abort(401, message="Bad token")
		self.log("%s autorizado" % u)
		return {"perfil": {
					'email': u.email,
					'nombre': u.nombre,
					'rut': u.rut,}
		}


class v2Upload(NewResource):
	"""Docstring"""
	LOG_TAG = "Upload"

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('token', required=False, location='headers')
		from werkzeug.datastructures import FileStorage
		parser.add_argument("photo", help="photo missing", type=FileStorage, required=True, location='files')
		parser.add_argument("height", help="height missing", required=True)
		args = parser.parse_args()
		token = args['token']
		photo = args['photo']
		height = args['height']

		self.log("Photo: %s, height: %s, token: %s" % (photo.filename, height, token))
		
		medidas, mensaje = tf.handle_photo(photo, height)
		
		if medidas is not None and token is not None:
			u = decodeToken(token)
			if u is not None:
				u.guardarMedidas(medidas)
			#self.log("Usuario no encontrado")
		
		return {"response": "upload", "medidas": medidas, "mensaje": mensaje}


class v2Register(NewResource):
	LOG_TAG = "Register"

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('email', help="email missing", required=True)
		parser.add_argument('pwd', help="pwd missing", required=True)
		parser.add_argument('nombre')
		parser.add_argument('rut')
		args = parser.parse_args()
		email = args['email']
		pwd = args['pwd']
		nombre = args['nombre']
		rut = args['rut']

		self.log("%s:%s:%s:%s" % (email, pwd, nombre, rut))

		u = Usuario.getUsuarioByEmail(email)
		if u is not None:
			self.log("Email ya existe")
			return abort(400, message="Email already exists")
		
		u = Usuario.addUsuario(email=email, pwd=pwd, nombre=nombre, rut=rut)
		token = makeToken(u)
		self.log("%s registrado" % u)		
		return {"response": "registrado", "token": token}


api.add_resource(v2Login, '/login')
api.add_resource(v2Perfil, '/perfil')
api.add_resource(v2Medidas, '/medidas')
api.add_resource(v2Upload, '/upload')
api.add_resource(v2Register, '/register')

######################
# Base de datos
# Uso de SQLite y SQLAlchemy
######################
from model import Usuario, initModel
from tf import tfManager

################################################
# TensorFlow
# Se instancia a la clase tfManager() para inicializar el modelo retinanet de deteccion de personas
# TF_ENABLED indica si es que se va a usar TF o no, para ganar tiempo en caso de debugging
# TODO Requests queue, since TF person_detector can't handle simultaneous requests
################################################
TF_ENABLED = 0
tf = tfManager(TF_ENABLED)
#from scripts.testpersona2 import a, b
#x = person_detector(photo)
#xx, yy = a()
#human = 'samples\\single2.jpg'

#from functools import wraps # TODO Implementar decoradores de auth

#initModel() # Si no existe el .sqlite, usar esta funcion para crear la BD segun los modelos definidos

# Tokens extremadamente basicos: solo la id del usuario
def makeToken(user):
	return str(user.id)


def decodeToken(token=None):
	if token is None: return None

	try:
		id = int(token)
		return Usuario.getUsuarioByID(id)
	except:
		return None


@app.errorhandler(404)
def page_not_found(e):
	return {"message": "404"}


#EOF
