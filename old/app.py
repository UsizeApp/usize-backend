from flask import Flask, session, redirect, url_for, escape, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import wtforms
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, IntegerField
from wtforms.validators import DataRequired
import flask_wtf, flask_wtf.file
import os

from time import sleep
#from functools import wraps # TODO Implementar decoradores de auth

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

from flask_restful import Resource, Api, reqparse, abort
api = Api(app)

# Logger
LOG = 1
DEFAULT_LOG = "Default"
LOGIN_LOG = "Login"

def logger(logtype=DEFAULT_LOG, logmsg=""):
	if LOG: print(">>> [%s] %s" % (logtype, logmsg))

parser = reqparse.RequestParser()
parser.add_argument('email', required=True, help="email")
parser.add_argument('pwd', required=True, help="pwd")

class v2Login(Resource):
	def get(self):
		return {'hello': 'world'}
	def post(self):
		args = parser.parse_args()
		email = args['email']
		pwd = args['pwd']

		logger(LOGIN_LOG, "%s:%s" % (email, pwd))

		u = Usuario.getUsuarioByEmail(email)
		if u is None:
			logger(LOGIN_LOG, "Email no existe")
			return abort(401, message="Bad login")
		if not u.validPassword(pwd):
			logger(LOGIN_LOG, "Password incorrecta")
			return abort(401, message="Bad login")
		logger(LOGIN_LOG, "%s autenticado" % u)		
		return {"response": "logueado", "token": str(u.id)}


api.add_resource(v2Login, '/v2/login')


################################################
# TensorFlow
# Se instancia a la clase tfManager() para inicializar el modelo retinanet de deteccion de personas
# TF_ENABLED indica si es que se va a usar TF o no, para ganar tiempo en caso de debugging
# TODO Requests queue, since TF person_detector can't handle simultaneous requests
################################################
TF_ENABLED = 0
from tf import tfManager
tf = tfManager(TF_ENABLED)
#from scripts.testpersona2 import a, b
#x = person_detector(photo)
#xx, yy = a()
#human = 'samples\\single2.jpg'


######################
# Base de datos
# Uso de SQLite y SQLAlchemy
######################
from model import initModel, Usuario
#initModel() # Si no existe el .sqlite, usar esta funcion para crear la BD segun los modelos definidos


################################################
# Rutas de Flask
################################################
WEB_MODE = 1 # Activar si vamos a hacer GETs y POSTs desde la web; desactivar si se deben devolver solo JSONs
IGNORE_FORM_VALIDATE = 1 # Activar para ignorar la validacion de las formas recibidas en POST

LOG_REQUESTS = 1
FAKE_TIMEOUT = 0

# Funcion para responder JSONs
def responder(s='null', d=None):
	sleep(FAKE_TIMEOUT)

	RESPUESTA = 'response'
	RESULTADO_BASE = 'null'
	RESPUESTA_BASE = {RESPUESTA: RESULTADO_BASE} # {'response': 'null'}

	resp = RESPUESTA_BASE

	# Si d es un diccionario, se usa como base
	if isinstance(d, dict):
		resp = d

	# Si s es un string, se usa como respuesta
	if isinstance(s, str):
		resp[RESPUESTA] = s # {'response': s}
			
	# Si el resultado final de resp es un diccionario, se retorna
	if isinstance(d, dict):
		j = jsonify(resp)
	else:
		j = jsonify(RESPUESTA_BASE)
	
	#if DBG: print(j.response)
	return j


@app.route('/')
def home():
	query = Usuario.query.all()
	return render_template('home.html', q=query) if WEB_MODE else 'Usize_GET_home'


class LoginForm(FlaskForm):
	defaultemail = None
	defaultpwd = None

	#if DBG:
		#defaultemail = 'example@email.com'
		#defaultpwd = 'holahola'

	email = StringField('email', validators=[DataRequired()])
	pwd = PasswordField('pwd', validators=[DataRequired()])


@app.route('/login', methods=['GET', 'POST'])
def login():
	form = LoginForm()

	if request.method == 'POST':
		if IGNORE_FORM_VALIDATE or form.validate_on_submit():
			email = form.email.data
			pwd = form.pwd.data

			if LOG_REQUESTS: print('>>> Login: %s %s' % (email, pwd))

			# Buscamos al usuario en la BD con el email
			u = Usuario.getUsuarioByEmail(email)
			# Si existe
			if u is not None:
				# Validamos la password con su hash
				if u.validPassword(pwd):
					if DBG: print('>>> /login: user_id %d logueado' % u.id)
					resul = 'logueado'
					token = makeToken(u)
					# Caso especial, retornamos de antemano con el token
					return responder(resul, {'token': token})
					
						#return redirect(url_for('home'))
				else:
					resul = 'pwd incorrecta'					
			else:
				resul = 'email no existe'
		else:
			if LOG_REQUESTS: print('>>> Login: Errors: ', form.errors)
			resul = 'bad_POST_request'
		
		return responder(resul)

	return render_template('login.html', form=form) if WEB_MODE else 'Usize_login'


class RegisterForm(FlaskForm):
	email = StringField('email', validators=[DataRequired()])
	pwd = PasswordField('pwd', validators=[DataRequired()])
	nombre = StringField('nombre', validators=[])
	rut = wtforms.IntegerField('rut', validators=[])


@app.route('/register', methods=['GET', 'POST'])
def register():
	form = RegisterForm()
	errors = None

	if request.method == 'POST':
		if IGNORE_FORM_VALIDATE or form.validate_on_submit():
			email = form.email.data
			pwd = form.pwd.data

			nombre = form.nombre.data
			rut = form.rut.data
			
			if LOG_REQUESTS: print('>>> Register: %s %s %s %d' % (email, pwd, nombre, rut))

			# Buscamos un posible usuario en la BD con el email
			u = Usuario.getUsuarioByEmail(email)
			# Si existe, entonces dicho email ya esta registrado
			if u is not None:
				errors = 'email ya existe'
			# Else, registramos al usuario
			else:
				resul = 'registrado'
				u = Usuario.addUsuario(email=email, pwd=pwd, nombre=nombre, rut=rut)
				token = makeToken(u)

				if WEB_MODE:
					return redirect(url_for('home'))
				else:
					return responder(resul, {'token': token})

				
	return render_template('register.html', form=form, errors=errors) if WEB_MODE else 'Usize_register_GET'


@app.route('/logout', methods=['GET'])
def logout():
	session.clear()
	return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
	if request.method == 'POST':
		token = request.form['token']

		if LOG_REQUESTS: print('>>> Profile POST: Token %s recibido' % token)

		u = decodeToken(token)
		if u is not None:
			if LOG_REQUESTS: print('>>> Profile POST: Usuario %d OK' % u.id)

			return responder('ok', {
				'email': u.email,
				'right_arm': u.right_arm,
				'left_arm': u.left_arm,
				'right_leg': u.right_leg,
				'left_leg': u.left_leg,
				'waist': u.waist,
				'hip': u.hip,
				'chest': u.chest,
				'bust': u.bust,
				'nombre': u.nombre,
				'rut': u.rut,
			})

		if LOG_REQUESTS: print('>>> Profile POST: Usuario no encontrado')
		return responder('user no encontrado')
	
	return 'Usize_profile_GET'


class UploadForm(FlaskForm):
	photo = flask_wtf.file.FileField('photo', validators=[flask_wtf.file.FileRequired()])
	height = wtforms.IntegerField('height', validators=[DataRequired()])


@app.route('/upload', methods=['GET', 'POST'])
def upload():
	form = UploadForm()

	if request.method == 'POST':
		if IGNORE_FORM_VALIDATE or form.validate_on_submit():
			photo = form.photo.data
			height = form.height.data
			#photo = request.files['photo']
			#photo_strea = photo.read()

			result = tf.handle_photo(photo, height)
			
			#if DBG: print(result)

			if isinstance(result, dict):
				# Si la sesion tiene un usuario, actualizamos las medidas
				if 'token' in request.form.keys():
					token = request.form['token']
					u = decodeToken(token)
					if u is not None: u.guardarMedidas(result)

				return jsonify(result)
			else:
				return jsonify({'result': 'fatal_error'})
		else:
			s = 'bad_POST_request'
			if DBG: print(s)
			return jsonify({'result': s})
	
	return render_template('upload.html', form=form) if WEB_MODE else 'Usize_GET_upload'


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

