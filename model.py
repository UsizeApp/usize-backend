from app import db

from werkzeug.security import generate_password_hash as genpw
from werkzeug.security import check_password_hash as checkpw

from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship

import datetime


# Email en Usize
# Genera una ID para el token
# Un Email puede tener varias Personas, y a lo menos una: la indicada en el registro
class Email(db.Model):
    __tablename__ = 'Emails'

    id = Column(Integer, primary_key=True)
    hash = Column(String(128), nullable=False)

    email = Column(String(80), unique=True, nullable=False)
    nombre = Column(String(80))
    rut = Column(Integer)

    # Indica si se debe mostrar la pantalla de bienvenida al iniciar sesion
    mostrar_bienvenida = Column(Boolean, default=False)

    # SQLAlchemy: colección de Personas
    personas = relationship("Persona")

    # Setea el hash de la contraseña del usuario
    def set_pwd(self, pwd):
        self.hash = genpw(pwd)

    # Valida que la contraseña recibida genere el mismo hash almacenado
    def validPassword(self, pwd):
        return checkpw(self.hash, pwd)

    # Obtiene los datos del email
    def getDatos(self):
        datos = {
            'email': self.email,
            'nombre': self.nombre,
            'rut': self.rut,

            'mostrar_bienvenida': self.mostrar_bienvenida,
            'personas': self.getPersonas()
        }

        return datos

    # Obtiene las Personas del email
    def getPersonas(self):
        personas = []
        for persona in self.personas:
            personas.append(str(persona.id))

        return personas

    ######################################################
    # Metodos estaticos
    @staticmethod
    def usuarioExample():
        if Email.query.count() <= 0:
            Email.addUsuario(email='ale@usm.cl', pwd='12345678',
                             nombre='Ale Sepulveda', rut=33333333, gender='M')

    # Retorna un Email si es que existe
    @staticmethod
    def buscarEmail(email):
        # MultipleResultsFound no deberia ocurrir
        return Email.query.filter_by(email=email.lower()).one_or_none()

    @staticmethod
    def getEmailByID(id):
        q = None
        try:
            q = Email.query.get(id)
        except Exception as e:
            print(e)

        return q

    # Constructor de Emails
    # Genera la Persona por defecto del email
    @staticmethod
    def addEmail(email, pwd, nombre='', rut=0, gender='M'):
        u = Email()
        u.set_pwd(pwd)

        u.email = email.lower()
        u.nombre = nombre
        u.rut = rut

        db.session.add(u)
        db.session.commit()

        p = Persona()
        p.alias = nombre if len(nombre) > 0 else 'Persona'
        p.gender = gender if gender in ['M', 'F'] else 'M'

        p.id_email = u.id

        db.session.add(p)
        db.session.commit()

        print("Email registrado con ID %d y Persona base %d" % (u.id, p.id))

        return u, p

    @staticmethod
    def all():
        return Email.query.all()


# Persona de Usize
# Controla las medidas calculadas
class Persona(db.Model):
    __tablename__ = 'Personas'

    id = Column(Integer, primary_key=True)

    # ID del email asociado
    id_email = Column(Integer, ForeignKey('Emails.id'))

    # Nombre de fantasia para la Persona
    alias = Column(String(80), default='Persona')
    # Genero de la persona
    gender = Column(String(1), default='M')

    # Medidas
    left_arm = Column(Integer, default=0)
    right_arm = Column(Integer, default=0)

    left_leg = Column(Integer, default=0)
    right_leg = Column(Integer, default=0)

    waist = Column(Integer, default=0)
    hips = Column(Integer, default=0)
    chest = Column(Integer, default=0)
    bust = Column(Integer, default=0)

    # Indica cuando se tomaron las ultimas medidas, ademas de saber si hay medidas o no
    fecha_ultimas_medidas = Column(DateTime)

    # Actualiza las medidas de la persona
    def guardarMedidas(self, medidas):
        self.left_arm = medidas['left_arm'] or 0
        self.right_arm = medidas['right_arm'] or 0

        self.right_leg = medidas['right_leg'] or 0
        self.left_leg = medidas['left_leg'] or 0

        self.waist = medidas['waist'] or 0
        self.hips = medidas['hips'] or 0
        self.chest = medidas['chest'] or 0
        self.bust = medidas['bust'] or 0

        self.fecha_ultimas_medidas = datetime.datetime.now()

        db.session.commit()

        return self.getDatos()

    # Obtiene los datos de la persona
    def getDatos(self):
        fecha_ultimas_medidas = self.fecha_ultimas_medidas
        
        if fecha_ultimas_medidas is not None:
            fecha_ultimas_medidas = fecha_ultimas_medidas.strftime("%d/%m/%Y %H:%M")
        datos = {
            'alias': self.alias,
            'gender': self.gender,
            'fecha_ultimas_medidas': fecha_ultimas_medidas,
            
            'medidas': self.getMedidas(),
        }

        return datos

    def getMedidas(self):
        medidas = {
            'left_arm': self.left_arm,
            'right_arm': self.right_arm,

            'left_leg': self.left_leg,
            'right_leg': self.right_leg,

            'waist': self.waist,
            'hips': self.hips,
            'chest': self.chest,
            'bust': self.bust,
        }

        return medidas

    @staticmethod
    def get(id):
        return Persona.query.get(id)
