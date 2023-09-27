from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import requests
import torch
from io import BytesIO
from PIL import Image
from typing import List
import cairosvg
import json
import re
from translator import TextTranslator


class ImageFeaturesExtraction:
    def __init__(self, threshold: float = 0.7, translate_source: str = "en", translate_target: str = "es"):
        self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.model.eval()
        self.transform = T.Compose([T.ToTensor()])
        self.threshold = threshold
        self.translator = TextTranslator(translate_source, translate_target)
        self.labels_url = 'https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt'
        self.coco_labels = self.load_coco_labels()

    def load_coco_labels(self):
        response = requests.get(self.labels_url)
        return response.text.splitlines()

    def translate_labels(self, labels):
        translations = [self.translator.translate(label) for label in labels]
        return [translation for translation in translations]

    def convert_svg_to_format(self, svg_data, output_format='png'):
        if output_format not in ['png', 'jpeg', 'pdf', 'ps', 'eps', 'svg']:
            raise ValueError("El formato de salida debe ser uno de: png, jpeg, pdf, ps, eps, svg")

        converted_data = cairosvg.svg2bytestring(bytestring=svg_data, output_format=output_format)
        return Image.open(BytesIO(converted_data))

    def analyze_news_images(self, image_urls: List[str], output_format='png'):
        results = []
        for img_url in image_urls:
            response = requests.get(img_url)

            # Comprobar si la imagen es SVG
            if 'image/svg+xml' in response.headers.get('content-type', ''):
                svg_data = response.content
                image = self.convert_svg_to_format(svg_data, output_format)
            else:
                image_extension = response.headers.get('content-type').split('/')[1]
                image = Image.open(BytesIO(response.content))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resto del código sigue igual
            transform = T.Compose([T.ToTensor()])
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(image_tensor)

            boxes = prediction[0]['boxes']
            scores = prediction[0]['scores']
            labels = prediction[0]['labels']

            filtered_boxes = boxes[scores > self.threshold]
            filtered_scores = scores[scores > self.threshold]
            filtered_labels = labels[scores > self.threshold]

            preprocessed_labels = [self.coco_labels[label.item()] for label in filtered_labels]

            detections = {}

            for label in preprocessed_labels:
                for score in filtered_scores.numpy():
                    clear_label = re.sub(r'^\d+:\s+', '', label)
                    final_label = self.translator.translate(clear_label)
                    detections[final_label] = float(score)

            results.append({
                'image_url': img_url,
                'detections': detections
            })

        return results

    def save_results_to_file(self, results, output_file='results.json'):
        with open(output_file, 'w') as file:
            json.dump(results, file, indent=4)
