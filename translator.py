from transformers import MarianMTModel, MarianTokenizer


class TextTranslator:
    def __init__(self, source_language, target_language):
        # Construir el nombre del modelo según los idiomas fuente y destino
        model_name = f"Helsinki-NLP/opus-mt-{source_language}-{target_language}"

        # Cargar el modelo y el tokenizador
        self.model = MarianMTModel.from_pretrained(model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

    def translate(self, text):
        # Tokenizar el texto de origen
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Realizar la traducción
        translation_ids = self.model.generate(input_ids)

        # Decodificar y obtener la traducción
        translated_text = self.tokenizer.decode(translation_ids[0], skip_special_tokens=True)

        return translated_text
