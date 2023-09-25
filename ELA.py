from PIL import Image, ImageChops, ImageEnhance
import requests
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
from typing import Tuple, List


class ELA:
    _image_extension: str = 'jpg'

    def __init__(self, image_url: str, download_path: str = 'temp_download', ela_image_path: str = 'temp_ela'):
        self._image_url = image_url
        self._download_path = download_path
        self._ela_image_path = ela_image_path

    def _download_image(self) -> Image:
        """Descarga una imagen desde una URL y devuelve una imagen de Pillow."""
        response = requests.get(self._image_url)
        # Verifica que la petición se haya completado exitosamente
        response.raise_for_status()

        self._image_extension = response.headers.get('content-type').split('/')[1]

        self._download_path = self._download_path + "." + self._image_extension
        self._ela_image_path = self._ela_image_path + "." + self._image_extension

        # Guarda la imagen en un archivo temporal
        with open(self._download_path, 'wb') as f:
            f.write(response.content)

        # Abre la imagen desde el archivo temporal
        image = Image.open(self._download_path)
        return image

    def _ela_analysis(self, image: Image, quality: int = 90, scale_factor: int = 10) -> Image:
        # Guarda una versión comprimida de la imagen usando una compresión JPEG específica
        temp_path = "temp_ela.jpg"
        image.save(temp_path, 'JPEG', quality=quality)

        # Recarga la versión comprimida
        compressed = Image.open(temp_path)

        # Calcula la diferencia absoluta entre la imagen original y la comprimida
        ela_image = ImageChops.difference(image, compressed)

        # Cierra la imagen original y la comprimida para liberar recursos
        image.close()
        compressed.close()

        # Amplifica las diferencias
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor * scale)

        # Guarda la imagen
        ela_image.save(self._ela_image_path)

        return ela_image  # Retorna la imagen ELA

    def _binarize_ela(self, ela_image: Image, threshold: int = 230) -> Image:
        """Convierte la imagen ELA en binario basándose en un umbral"""
        gray_image = ela_image.convert('L')
        return gray_image.point(lambda p: 255 if p > threshold else 0)

    def _detect_edited_regions(self, binarized_ela):
        """Detecta y etiqueta las regiones en la imagen binarizada"""
        labeled, num_features = ndimage.label(binarized_ela)
        return labeled, num_features

    def _estimate_edit_probability(self, labeled_regions_array: np.ndarray) -> int:
        total_pixels = labeled_regions_array.size
        edited_pixels = (labeled_regions_array > 0).sum()
        return edited_pixels / total_pixels

    def _adaptive_threshold(self, ela_image: Image, percentile: int = 95) -> int:
        """
        Calcula un umbral adaptativo basado en un percentil de la imagen ELA.
        """
        ela_data = np.array(ela_image.convert('L'))  # Convierte a escala de grises
        threshold = np.percentile(ela_data, percentile)
        return threshold

    def show_original_image(self):
        # Mostrar la imagen
        img = plt.imread(self._download_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.colorbar()
        plt.axis('off')  # Desactiva los ejes de coordenadas
        plt.show()

    def show_ela_image(self):
        # Mostrar la imagen
        img = plt.imread(self._ela_image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.colorbar()
        plt.axis('off')  # Desactiva los ejes de coordenadas
        plt.show()

    def visualize_regions(self, labeled_regions_array: np.ndarray) -> None:
        """Visualiza las regiones etiquetadas con diferentes colores."""
        plt.figure(figsize=(10, 10))

        # Usamos 'nipy_spectral' para un mejor contraste en la visualización
        plt.imshow(labeled_regions_array, cmap='nipy_spectral')

        plt.colorbar()
        plt.axis('off')
        plt.show()

    def calculate_estimated_edit_percentaje(self) -> Tuple[float, int, List[int]]:
        image = self._download_image()
        ela_img = self._ela_analysis(image)

        threshold = self._adaptive_threshold(ela_img)
        binarized = self._binarize_ela(ela_img, threshold)
        labeled_regions_response, num_of_regions_response = self._detect_edited_regions(np.array(binarized))

        probability = self._estimate_edit_probability(labeled_regions_response)
        return probability, num_of_regions_response, labeled_regions_response
