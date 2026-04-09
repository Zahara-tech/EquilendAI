import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_database():
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    return client["equilend_ai"]

def insert_application(data):
    db = get_database()
    collection = db["applications"]
    result = collection.insert_one(data)
    return result.inserted_id

def fetch_all_applications():
    db = get_database()
    collection = db["applications"]
    return list(collection.find({}, {"_id": 0}))