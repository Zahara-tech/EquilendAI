from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")
print("URI =", uri)   # 👈 VERY IMPORTANT

client = MongoClient(uri)
db = client["equilend_ai"]

db.test.insert_one({"msg": "connected"})
print("MongoDB Connected Successfully!")