models = {'linear_svm': SVC(C = 1, kernel='linear', gamma='scale', probability=True),
          'knn': KNeighborsClassifier(n_neighbors=5),
          'RF': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=2, n_estimators=600),
          'Logistic': LogisticRegression(),
          'XGB':XGBClassifier(),
          'LGBM': LGBMClassifier(),
          'BLGBM': BaggingClassifier(base_estimator=LGBMClassifier() , n_jobs=-1, n_estimators = 15)}}
# Create scores dictionary for each algorithm
scores=[]
mdl=[]
results=[]
for model in models.keys():
    est = models[model]
    est.fit(X_train,  y_train)
    mdl.append(model)
    y_pred = est.predict(X_test)
    results.append((est.score(X_test, y_test), y_pred))
    #print(metrics.classification_report(y_test, y_pred, digits=3))
results = [dict(zip(['Accuracy','y_pred'], i)) for i in results]

# At this point "scores" only contains accuracy and y_pred for each one of the best models chosen for each algorithm
scores = dict(zip(mdl, results))

# Enrich scores dictionary with the extra metrics
minRe={}
maxRe={}
from sklearn.metrics import precision_recall_fscore_support as classmetrics
for alg in scores.keys():
    print ("\n",alg)
    precision, recall, fscore, support = classmetrics(y_test,scores[alg]['y_pred'])
    print ('Recall   : {}'.format(recall))
    scores[alg]['Precision'] = precision
    scores[alg]['Recall'] = recall
    scores[alg]['F1'] = fscore
    scores[alg]['Support']  = support
    minRe[alg] = np.argmin(scores[alg]['Recall'])
    maxRe[alg] = np.argmax(scores[alg]['Recall'])
print("\nWorst performance class for each classifier based on Recall:")
print(maxRe)
print("Best performance class for each classifier based on Recall:")
print(minRe)