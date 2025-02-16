from pymongo import MongoClient
import pandas as pd
import pandas

dataset=pd.read_excel("dining_info.xlsx")
dataset

client = MongoClient("mongodb+srv://sonyeda601:buodR9tHY0aIjd4A@cluster0.gbw0x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["hotel_guests"]

df = pd.read_excel('dining_info.xlsx')
df.drop('Unnamed: 0',axis=1,inplace=True)
collection = db["dining_info"]
collection.insert_many(df.to_dict(orient="records"))