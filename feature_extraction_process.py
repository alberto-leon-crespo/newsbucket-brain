import json
from image_features_extraction import ImageFeaturesExtraction
from pymongo import MongoClient
from decouple import config
from urllib.parse import urlparse, urlunparse

mongo_uri = config('MONGO_URI')
database_name = 'newsbucket'
collection_name = 'news'  # Cambia esto al nombre de tu colección

noticias_analyzer = ImageFeaturesExtraction()
noticias_analyzer_db = MongoClient(mongo_uri)

parsed_url = urlparse(mongo_uri)

# Crear una tupla de elementos de la URL sin el usuario y la contraseña
url_tuple = (parsed_url.scheme, parsed_url.netloc.split('@')[-1], parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment)

# Reconstruir la URL sin el usuario y la contraseña
new_connection_string = urlunparse(url_tuple)

print("Sucessful connected to database " + str(new_connection_string))

database = noticias_analyzer_db[database_name]  # Selecciona la base de datos
collection = database[collection_name]  # Selecciona la colección

news_documents = collection.find()

for news_doc in news_documents:
    if "imgs" in news_doc:
        img_urls = news_doc["imgs"]
        if len(img_urls) > 0:
            print("Extacting features from images " + json.dumps(img_urls))
            img_results = noticias_analyzer.analyze_news_images(img_urls)

            # Actualizar el documento de noticias con los resultados
            update_data = {}
            for i, result in enumerate(img_results, start=1):
                field_name = f'img{i}'
                update_data[field_name] = result['detections']

            # Identificar el documento a actualizar por su _id
            filter_query = {"_id": news_doc["_id"]}

            print(update_data);

            # Realizar la actualización utilizando update_one()
            collection.update_one(filter_query, {"$set": update_data})

            print("Processed new with id #" + str(news_doc["_id"]))

print("Procesamiento de noticias completado.")