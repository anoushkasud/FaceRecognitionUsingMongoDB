from pymongo import MongoClient
from deepface import DeepFace
from tqdm import tqdm
import os
import pandas as pd


connection = 'mongodb://localhost:27017/'
client = MongoClient(connection)
database = 'AttendanceSystem'; collection = 'deepface'
db = client[database]

folderPath = "images"
facial_img_paths = os.listdir(folderPath)
 
instances = []
for i in tqdm(range(0, len(facial_img_paths))):
    facial_img_path = 'images' + "/" + facial_img_paths[i]    
    embedding = DeepFace.represent(img_path = facial_img_path, model_name = "Facenet")[0]["embedding"]
     
    instance = []
    instance.append(facial_img_path[7:-4])#
    instance.append(embedding)
    instances.append(instance)


df = pd.DataFrame(instances, columns = ["studentId", "embedding"])

for index, instance in tqdm(df.iterrows(), total = df.shape[0]):
    db[collection].insert_one({"studentId": instance["studentId"], "embedding" : instance["embedding"]})
