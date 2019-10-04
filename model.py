from app import db

from werkzeug.security import generate_password_hash as genpw
from werkzeug.security import check_password_hash as checkpw

class Usuario(db.Model):
	__tablename__ = 'Usuarios'

	id = db.Column(db.Integer, primary_key=True)
	email = db.Column(db.String(80), unique=True, nullable=False)
	hash = db.Column(db.String(128), nullable=False)

	right_arm = db.Column(db.String(10))
	left_arm = db.Column(db.String(10))
	right_leg = db.Column(db.String(10))
	left_leg = db.Column(db.String(10))
	waist = db.Column(db.String(10))
	hip = db.Column(db.String(10))
	chest = db.Column(db.String(10))
	bust = db.Column(db.String(10))

	nombre = db.Column(db.String(80))
	rut = db.Column(db.Integer)
	gender = db.Column(db.String(10))

	# Setea el hash de la contraseña del usuario
	def set_pwd(self, pwd):
		self.hash = genpw(pwd)

	# Valida que la contraseña recibida genere el mismo hash almacenado
	def validPassword(self, pwd):
		return checkpw(self.hash, pwd)

	# Actualiza las medidas del usuario
	def guardarMedidas(self, medidas):
		self.right_arm = medidas['right_arm'] or 0
		self.left_arm = medidas['left_arm'] or 0
		self.right_leg = medidas['right_leg'] or 0
		self.left_leg = medidas['left_leg'] or 0
		self.waist = medidas['waist'] or 0
		self.hip = medidas['hip'] or 0
		self.chest = medidas['chest'] or 0
		self.bust = medidas['bust'] or 0

		db.session.commit()
	
	####################################
	# Metodos estaticos de los Usuarios
	####################################

	@staticmethod
	def usuarioExample():
		if Usuario.query.count() <= 0:
			Usuario.addUsuario(email='example@email.com', pwd='holahola')

	@staticmethod
	def getUsuarioByEmail(email):
		q = None
		email = email.lower()
		try:
			q = Usuario.query.filter_by(email=email).one_or_none()
		#except Exception as e: # MultipleResultsFound
		except: # MultipleResultsFound
			#print(e)
			print("WARNING: email %s duplicado" % email)
		
		return q

	@staticmethod
	def getUsuarioByID(id):
		q = None
		try:
			q = Usuario.query.get(id)
		except:
			print(e)
		
		return q

	@staticmethod
	def addUsuario(email, pwd, nombre='', rut=0, gender=None):
		u = Usuario(email=email.lower())
		u.set_pwd(pwd)

		u.nombre = nombre
		u.rut = rut
		u.gender = gender

		db.session.add(u)
		db.session.commit()
		return u

	@staticmethod
	def all():
		return Usuario.query.all()


# Usar si es que se quiere empezar la BD desde cero
def initModel():
	db.create_all()
	db.session.commit()
	Usuario.usuarioExample()

