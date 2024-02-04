"""
	Preparación de Data 4-11
"""
def prepareData(dataset, val_size=0.20):
	vector = dataset.values
	X = vector[:, 0:dataset.shape[1] - 1].astype(float)
	Y = vector[:, dataset.shape[1] - 1]
	return train_test_split(X, Y, test_size=val_size)
X_train, X_test, Y_train, Y_test = prepareData(dataset, 0.1)
"""
	Modelos de Prueba
"""
models = []
models.append(('Árbol de Decisión', DecisionTreeClassifier()))
models.append(('k Vecinos + Cercanos', KNeighborsClassifier()))
models.append(('Máquina de Soporte Vectorial', SVC()))
"""
	Evaluación del Algoritmo
"""
def evaluateModels(models, num_folds, scoring, X_train, Y_train):
	results = {}
	for name, model in models:
		kfold = KFold(n_splits=num_folds)
		cv_results = cross_val_score
			(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results[name] = cv_results
	return pd.DataFrame(data=results)
resultados = evaluateModels(models, 10, 'accuracy', X_train, Y_train)
"""
	Comparación de Algoritmos
"""
def compareResults(results):
	new_result = pd.concat([results.mean(), results.std()], axis=1)
	new_result.columns = ["Precisión media", "Desviación Estandar"]
	sns.boxplot(data=results, orient='h')
	pyplot.xlim(xmin=0.7, xmax=1)
	pyplot.show()
	return new_result
"""
	Normalize the dataset
"""
def normalizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('Norm' + name, Pipeline
			([('Normalizer', Normalizer()), (name, model)])))
	return pipelines
"""
	Standardize the dataset
"""
def standarizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('Scaled' + name, Pipeline
			([('Scaler', StandardScaler()), (name, model)])))
	return pipelines
"""
	Validacion de Resultados
"""
def validateModels(models, X_train, Y_train, X_validation, Y_validation):
	results = {'Modelo':[], 'Precisión':[], 'Error':[], 'cf_matrix':[]}
	for name, model in models:
		model.fit(X_train, Y_train)
		Y_predicted = model.predict(X_validation)
		results['Modelo'].append(name)
		results['cf_matrix'].append(
			confusion_matrix(y_pred=Y_predicted, y_true=Y_validation))
		results['Precisión'].append(
			accuracy_score(y_pred=Y_predicted, y_true=Y_validation))
		results['Error'].append(
			mean_squared_error(y_pred=Y_predicted, y_true=Y_validation))
	return pd.DataFrame(data=results)
"""
	Class de Selección de Estimadores - función fit
"""
def fitSearch(self, X, y, cv=KFold(n_splits=5), n_jobs=1, verbose=1, 
		scoring=None, **kwargs):
	for key in self.keys:
		print("Running Search for %s." % key)
		start = time.time()
		gs = Method_SearchCV(self.models[key], self.params[key], cv=cv, 
				n_jobs=n_jobs, verbose=verbose, scoring=scoring,
				refit=refit, **kwargs)
		self.grid_searches[key] = gs.fit(X, y)
		end = time.time()
		self.time_model[key] = [end-start]
"""
	Class de Selección de Estimadores - función score
"""
def score(self):
	for estimator in self.keys:
		d = {}
		d['.Accuracy'] = self.grid_searches[estimator].cv_results_['mean_test_score']
		d['.Error'] = self.grid_searches[estimator].cv_results_['std_test_score']
		df1 = pd.DataFrame(d)
		d = self.grid_searches[estimator].cv_results_['params']
		df2 = pd.DataFrame(list(d))
		self.result[estimator] = pd.concat([df1, df2], axis=1).fillna(' ')
	return pd.concat(self.result).fillna(' ')

"""
	Tune scaled Model
"""
def tuneScaledModel(model, X_train, Y_train, 
				num_folds, seed, param_grid, scoring):
	scaler = StandardScaler().fit(X_train)
	rescaledX = scaler.transform(X_train)
	kfold = KFold(n_splits=num_folds, random_state=seed)
	grid = GridSearchCV(estimator=model, 
		param_grid=param_grid, scoring=scoring, cv=kfold)
	grid_result = grid.fit(rescaledX, Y_train)
	return grid_result


# Show Tune Results
def tuneResults(grid_result):
	print("\nBest: %f using %s" % 
		(grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))






























































































