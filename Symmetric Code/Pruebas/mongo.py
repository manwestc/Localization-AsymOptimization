from pymongo import MongoClient
from datetime import datetime
from pprint import pprint
from pandas import DataFrame
from matplotlib import pyplot
"""
mongoimport --db sensoresDB --collection seminario --file ./backupDataM.json
> use sensoresDB
> db.seminario.find().pretty()

{
	"_id" : ObjectId("58990ee8af4b2479c92b65f8"),
	"id_moduleiot" : "C002",
	"id_sensor" : "S50002",
	"type" : "Temperatura",
	"value" : 32.5,
	"date" : "2016-04-10",
	"hour" : "16:16:15"
}

"""

client = MongoClient('localhost', 27017)
db = client.sensoresDB
collection = db.seminario
# pprint(collection.find_one())
full = DataFrame(list(collection.find()))
temperatura = DataFrame(list(collection.find({"$or": [{"type": "Temperatura"}, {"type": "temperatura"}]})))
monoxido = DataFrame(list(collection.find({"$or": [{"type": "Monoxido"}, {"type": "monoxido"}]})))
dioxido = DataFrame(list(collection.find({"$or": [{"type": "Dioxido"}, {"type": "dioxido"}]})))
presion = DataFrame(list(collection.find({"$or": [{"type": "Presion"}, {"type": "presion"}]})))

temperatura.describe()  # 3862 datos
monoxido.describe()		# 4 datos
dioxido.describe()		# 7 datos
presion.describe()		# 3867 datos

temperatura.plot(kind='density', subplots=True, sharex=False)
presion.plot(kind='density', subplots=True, sharex=False)
pyplot.show()

full.groupby('type').size()
"""
type
Dioxido           7
Monoxido          4
Presion        3804
Temperatura    3803
presion          63
temperatura      59
"""