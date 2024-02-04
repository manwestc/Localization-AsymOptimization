-- latex informe Seminario 
estado del arte
	seguridad ciudadana precobs
	meteorologia salud ..
	aplicaciones de machine learning para analisis
sklearn y algoritmos
pruebas y resultados
	data limpia csv / data perfecta
	data de mongodb beagons, sensores
conclusiones y trabajos futuros

"                         Model      Mean       STD\n",
"7           AdaBoostClassifier  0.798338  0.048969\n",
"9   GradientBoostingClassifier  0.742983  0.060308\n",
"3         KNeighborsClassifier  0.742780  0.072794\n",
"6         ExtraTreesClassifier  0.736634  0.116844\n",
"8       RandomForestClassifier  0.709850  0.117016\n",
"1                          SVC  0.683069  0.056990\n",
"4       DecisionTreeClassifier  0.675291  0.067551\n",
"2                   GaussianNB  0.594325  0.035160\n",
Parametric Machine Learning Algorithms
	- Simpler, Speed, Less D
"5           LogisticRegression  0.545059  0.049457\n",
"0 Linear Discriminant Analysis  0.527599  0.053464\n",

"CPU times: user 18min 21s, sys: 6.67 s, total: 18min 28s\n",
"Wall time: 18min 30s\n"

	ExtraTreesClassifier, RandomForestClassifier, 
	AdaBoostClassifier, GradientBoostingClassifier
	

	StandardScaler
	ExtraTreesClassifier
models
	ExtraTreesClassifier: pipeline,
	RandomForestClassifier: RandomForestClassifier(),
	AdaBoostClassifier: AdaBoostClassifier(),
	GradientBoostingClassifier: GradientBoostingClassifier(),
	SVC: SVC()

#STEP 0: Define workflow parameters\n",
"definer = define.Define(nameData=name, className=className).pipeline()\n",
"\n",
#STEP 1: Analyze data by ploting it\n",
"#analyze.Analyze(definer).pipeline()\n",
"\n",
#STEP 2: Prepare data by scaling, normalizing, etc. \n",
"preparer = prepare.Prepare(definer).pipeline()\n",
"\n",
#STEP 3: Feature selection\n",
"featurer = feature_selection.FeatureSelection(definer).pipeline()\n",
"\n",
#STEP4: Evalute the algorithms by using the pipelines\n",
"evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()\n",

"""
use seminario
db.pruebas.find({"type":"Temperatura"}).pretty()
db.pruebas.find({"type":"Monoxido"}).pretty()
db.pruebas.find({"type":"Dioxido"}).pretty()
db.pruebas.find({"type":"Presion"}).pretty()

{
	"_id" : ObjectId("58a217bc634fd20ac2e2faa8"),
	"id_moduleiot" : "C002",
	"id_sensor" : "S55002",
	"value" : -10,
	"type" : "temperatura",
	"date" : "2017-02-13",
	"hour" : "15:31:56"
}


db.pruebas.find({
		$and:[
			{"type":{$ne:"Temperatura"}},
			{"type":{$ne:"Dioxido"}},
			{"type":{$ne:"Monoxido"}},
			{"type":{$ne:"Presion"}}
		]
	}).pretty()

db.pruebas.update({"type":"presion"},
	{$set:{"type":"Presion"}}, {multi:true})


> db.pruebas.find({"type":"Presion"}).count()
3867
> db.pruebas.find({"type":"Temperatura"}).count()
3862
> db.pruebas.find({"type":"Dioxido"}).count()
7
> db.pruebas.find({"type":"Monoxido"}).count()
4

"""

Te adjunto los algoritmos qué estamos utilizando y algunos detalles:

	Donde dice Voting(GBC-ET), es el algoritmo Voting, el cual toma como parametros dos algoritmos: el GradientBostingClassifier y el ExtraTrees
	El voting pondera esos dos algoritmos y saca la mejor prediccion de dicha ponderacion.
	LDA: Linear Discriminant Analysis
	C-Support Vector
	El ultimo es un caso particular de SVM.
	Y por ultimo todos los algoritmos estan aqui: http://scikit-learn.org/stable/modules/classes.html. Toda esta info detallada se irá poniendo en la documentacion de pymach que es nuestra librería de Machine Learning
	Estos algoritmos solo son de la parte evaluation.py de dicha librería
	Obviamente hay más en los pasos anteriores: PCA, Scaling, etc..

	Te adjunto el enlace del libro de Machine learning que estamos utizando en Python. Está explicada muy bien la teoría sin tanta teoría matemática. Debes poner el foco en este orden de estudio:
		Machine Learning Algorithms (Teoría)
		MAchine Learning algorithms from Scratch (Ejemplos de implementacion)
		Machine Learning Mastery with Python (Más avanzado que el enlace anterior)



from pymongo import MongoClient
from datetime import datetime
import pymongo

client = MongoClient()
#client = MongoClient("mongodb://mongodb0.example.net:27019")
db = client.test
#db = client['primer']
coll = db.dataset
#coll = db['dataset']

print coll

cursor = db.pruebas.find()
for document in cursor:
  print(document)
print "meow"

# buscar un objeto
cursor = db.pruebas.find({"type": "Temperatura"})
for document in cursor:
  print(document)

# para buscar un objeto con su subobjeto
cursor = db.pruebas.find({"date": "2017-02-13"})
for document in cursor:
  print(document)

# para buscar un objeto en un array
cursor = db.pruebas.find({"value": {$gt:"15"}})
for document in cursor:
  print(document)

# para filtrar datos
cursor = db.pruebas.find({"grades.score": {"$gt": 30}})
for document in cursor:
  print(document)

# AND
cursor = db.pruebas.find({"cuisine": "Italian", "address.zipcode": "10075"})

# OR
cursor = db.pruebas.find({"$or": [{"cuisine": "Italian"}, {"address.zipcode": "10075"}]})

# SORT
cursor = db.pruebas.find().sort([
	("borough", pymongo.ASCENDING),
	("address.zipcode", pymongo.ASCENDING)
])

# update el primero con las condiciones de name = Juni
result = db.pruebas.update_one(
	{"name": "Juni"},
	{
		"$set": {
			"cuisine": "American (New)"
		},
		"$currentDate": {"lastModified": True}
	}
)

# embedded field
result = db.pruebas.update_one(
	{"restaurant_id": "41156888"},
	{"$set": {"address.street": "East 31st Street"}}
)

# update todos con las condiciones de cuisine:other y address.zipcode=10016
result = db.pruebas.update_many(
	{"address.zipcode": "10016", "cuisine": "Other"},
	{
		"$set": {"cuisine": "Category To Be Determined"},
		"$currentDate": {"lastModified": True}
	}
)


# reemplazar toda una fila (o documento)
result = db.pruebas.replace_one(
	{"_id": ObjectId("58a217bc634fd20ac2e2faa8")},
	{
		"id_moduleiot": "C002",
		"id_sensor": "S55002",
		"value": -10,
		"type": "temperatura",
		"date": "2017-02-13",
		"hour": "15:31:56"
	}
)

# borrar
result = db.pruebas.delete_many({"id_sensor": "S55002"})

# borrar todo
result = db.pruebas.delete_many({})

# eliminar base datos
db.pruebas.drop()


