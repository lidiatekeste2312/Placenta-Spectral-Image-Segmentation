import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import joblib
import h5py

##### load dataset of selected samples
data= h5py.File('OneDrive_1_3-13-2021/dataset_selected.hdf5','r')
n1 = data.get('dset_train')

n2 = np.array((n1.shape[0],n1.shape[1]))
n2 = np.array(n1).copy()
np.random.shuffle(n2)

###train gradient boosting scikit-learn model
model_gd = GradientBoostingClassifier(random_state=1, verbose= True)
X = n2[:,:-1]  
y = n2[:,-1]
X = np.array(X)
y = np.array(y).astype(int)
model_gd.fit(X,y)
GD_model = joblib.dump(model_gd, 'OneDrive_1_3-13-2021/GD_model.pkl')
##############################################################
########extra models trained#############################
# model_rf = RandomForestClassifier(random_state=1, verbose= True)
# X = n2[:,:-1]  
# y = n2[:,-1]
# X = np.array(X)
# y = np.array(y).astype(int)
# model_rf.fit(X,y)
# RF_model = joblib.dump(model_rf, 'OneDrive_1_3-13-2021/RF_model.pkl')

# model = MLPClassifier(random_state=1, verbose= True, early_stopping=True)
# classes = [0,1,2,3,4,5,6,7,8,9,10]
# X = n2[:,:-1]  
# y = n2[:,-1]
# X = np.array(X)
# y = np.array(y).astype(int)
# model.fit(X,y)
# MLP2_model = joblib.dump(model, 'OneDrive_1_3-13-2021/MLP2_model.pkl')

# svm =  SVC(C = 100, kernel = 'rbf', cache_size = 10*1024, verbose=True)
# X = n2[:,:-1]  
# y = n2[:,-1]
# X = np.array(X)
# y = np.array(y).astype(int)
# svm.fit(X,y)
# svm_model = joblib.dump(svm, 'OneDrive_1_3-13-2021/svm_model.pkl')
