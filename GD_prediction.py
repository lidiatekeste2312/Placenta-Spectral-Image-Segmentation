import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from spectral_tiffs import read_mtiff, read_stiff, write_mtiff, write_stiff
import numpy as np
from skimage import color
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import jaccard_score

def analyze(image):
    model1 = joblib.load('C:/Users/Admin/Documents/idp/OneDrive_1_3-13-2021/Python-scripts/gd_model.pkl')


    Blue_dye = [0,255,255]
    Red_dye = [128,0,0]
    ICG = [0,255,0]
    Stroma_ICG= [255,255,255]
    Stroma = [128,128,128]
    Umbilical_cord = [255,242,0]
    Suture = [128,128,0]
    Artery_ICG = [34,177,76]
    Vein = [63,72,204]
    Artery = [237,28,36]
    non = [0,0,0]
    colors=[non, Blue_dye, Red_dye, ICG, Stroma_ICG, Stroma, Umbilical_cord,
            Suture, Artery_ICG, Vein, Artery]
    lbl_handles=["non", "Blue_dye", "Red_dye", "ICG", "Stroma_ICG", "Stroma", "Umbilical_cord",
            "Suture", "Artery_ICG", "Vein", "Artery"]
    v = np.zeros((1024,1024,3))
    
    
    
    img = "C:/Users/Admin/Documents/idp/OneDrive_1_3-13-2021/prediction/upper_6_icg.tif"
    #img = image
    spim,w,rgb,meta = read_stiff(img)
    
    sx = []
    for i in range (0,1024):
      for j in range (0,1024):
        temp = []
        for z in range (0,38):
          temp.append(spim[i,j,z])
        sx.append(temp)
        
    
    
    
    prediction = model1.predict(np.array(sx))
    prediction = prediction.reshape((1024,1024))
    for i in range (prediction.shape[0]):
        for j in range (prediction.shape[1]):
            if (prediction[i,j]>0):
                v[i,j]=colors[prediction[i,j]]
            else:
                v[i,j]=rgb[i,j]
    v = v/255
    labels = np.unique(prediction)
    patches = []
    for i in range(labels.shape[0]):
        c = np.array((colors[labels[i]]))/255
        patch= mpatches.Patch(color=c, label=lbl_handles[labels[i]])
        patches.append(patch)
    px = 1/plt.rcParams['figure.dpi']
    plt.figure(figsize=(512*px, 512*px))
    plt.imshow(v)
    plt.legend(handles=patches, fontsize='small',loc='upper left',bbox_to_anchor=(1, 1))
    #plt.show()
    path = img+'_mask.png'
    plt.savefig(path,bbox_inches="tight")

    return path
