from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas


st.set_page_config(
	page_title="Predictor de prendas",
	page_icon="👕",
	layout="centered",
)

CLASS_NAMES = [
	"T-shirt/top",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
]
MODEL_PATH = Path(__file__).with_name("prendas_model.keras")


@st.cache_resource
def load_model():
	try:
		model = tf.keras.Sequential(
			[
				tf.keras.layers.Input(shape=(28, 28)),
				tf.keras.layers.Flatten(),
				tf.keras.layers.Dense(64, activation="relu"),
				tf.keras.layers.Dense(32, activation="relu"),
				tf.keras.layers.Dense(16, activation="relu"),
				tf.keras.layers.Dense(10, activation="softmax"),
			]
		)
		model.load_weights(MODEL_PATH)
		return model
	except (ValueError, TypeError) as error:
		raise RuntimeError(
			"No se pudieron cargar los pesos de prendas_model.keras. "
			"Verifica que el archivo del modelo esté en el repositorio."
		) from error


def prepare_image(image: Image.Image) -> np.ndarray:
	grayscale = ImageOps.grayscale(image)
	resized = grayscale.resize((28, 28), Image.Resampling.LANCZOS)
	pixels = np.asarray(resized, dtype=np.float32) / 255.0
	return pixels


def predict(image: Image.Image):
	model = load_model()
	pixels = prepare_image(image)
	probabilities = model.predict(pixels[np.newaxis, ...], verbose=0)[0]
	predicted_index = int(np.argmax(probabilities))
	return pixels, predicted_index, probabilities


st.title("Predictor de prendas con TensorFlow")
st.write("Dibuja una prenda o carga una imagen para clasificarla con Fashion-MNIST.")

input_mode = st.radio("Origen de la imagen", ["Dibujar", "Subir imagen"], horizontal=True)
image = None

if input_mode == "Dibujar":
	try:
		canvas_result = st_canvas(
			fill_color="rgba(255, 255, 255, 1)",
			stroke_width=12,
			stroke_color="#FFFFFF",
			background_color="#000000",
			height=280,
			width=280,
			drawing_mode="freedraw",
			key="clothing_canvas",
		)
		if canvas_result.image_data is not None:
			image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
	except RuntimeError:
		st.error(
			"El canvas no está disponible en este despliegue. "
			"Puedes cambiar a 'Subir imagen' para continuar."
		)
else:
	uploaded_file = st.file_uploader(
		"Selecciona una imagen",
		type=["png", "jpg", "jpeg", "webp"],
	)
	if uploaded_file is not None:
		image = Image.open(uploaded_file)
		st.image(image, caption="Imagen cargada", width=280)

if st.button("Predecir prenda", type="primary", use_container_width=True):
	if image is None:
		st.warning("Dibuja algo en el canvas o carga una imagen antes de predecir.")
	else:
		try:
			with st.spinner("Analizando la imagen..."):
				processed_image, predicted_index, probabilities = predict(image)

			st.success(f"Predicción: {CLASS_NAMES[predicted_index]}")
			st.image(
				processed_image,
				caption="Entrada procesada a 28 x 28 píxeles",
				width=224,
			)

			top_indices = np.argsort(probabilities)[-3:][::-1]
			st.subheader("Probabilidades")
			for index in top_indices:
				st.write(f"{CLASS_NAMES[index]}: {probabilities[index]:.1%}")
				st.progress(float(probabilities[index]))
		except RuntimeError as error:
			st.error(str(error))

st.divider()
st.subheader("Instrucciones")
st.markdown(
	"""
	- En el canvas, dibuja una sola prenda centrada sobre el fondo negro.
	- El lápiz tiene un ancho medio y la aplicación convierte el dibujo a 28 x 28 píxeles.
	- Las imágenes se convierten a escala de grises y se normalizan dividiendo sus píxeles entre 255.
	- Para obtener mejores resultados, carga imágenes similares a las imágenes de Fashion-MNIST usadas durante el entrenamiento: prendas aisladas, con fondo negro y la prenda clara hacia el blanco.
	- El modelo devuelve probabilidades mediante una capa de salida `softmax`.
	"""
)
