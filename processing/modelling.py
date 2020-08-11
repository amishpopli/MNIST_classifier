from processing.utils import getData
from processing.utils import augment
from processing.utils import saveOutput
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

### Read data
mnist = getData()
data, target = mnist["data"], mnist["target"]

### startified split
train_x, train_y, test_x, test_y = train_test_split(data, target,stratify=target, test_size=0.25)

### Build baseline model
model = KNeighborsClassifier()
model.fit(train_x,test_x)
predictions = model.predict(train_y)
### accuracy
accuracy_baseline = accuracy_score(test_y,predictions)

#### grid search
model = KNeighborsClassifier()
params_grid = {"weights":["uniform","distance"],"n_neighbors":[3,5,10]}
grid = GridSearchCV(model, params_grid, cv=5, verbose=3)
grid.fit(train_x,test_x)
predictions = grid.predict(train_y)
### accuracy_grid
accuracy_grid_search = accuracy_score(test_y,predictions)

### data augment
model = KNeighborsClassifier()
train_x,test_x = augment(train_x,test_x)
model.fit(train_x,test_x)
predictions = model.predict(train_y)
### accuracy
accuracy_augment = accuracy_score(test_y,predictions)

#### grid search
model = KNeighborsClassifier()
params_grid = [{"weights":["uniform","distance"],"n_neighbors":[3,5,10]}]
grid = GridSearchCV(model, params_grid, cv=5, verbose=3)
grid.fit(train_x,test_x)
predictions = grid.predict(train_y)
### accuracy_grid
accuracy_augment_grid_search = accuracy_score(test_y,predictions)

### save output to txt file
saveOutput({"accuracy_baseline":accuracy_baseline,\
            "accuracy_grid_search":accuracy_grid_search,\
            "accuracy_augment":accuracy_augment,\
            "accuracy_augment_grid_search": accuracy_augment_grid_search})