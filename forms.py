from flask_wtf import FlaskForm
from wtforms import StringField, DateField, TimeField, SubmitField
from wtforms.validators import DataRequired

class EventForm(FlaskForm):
    nombre = StringField('Nombre', validators=[DataRequired()])
    fecha = DateField('Fecha', validators=[DataRequired()])
    hora = TimeField('Hora', validators=[DataRequired()])
    submit = SubmitField('Agregar Evento')
