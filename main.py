import joblib
from skimage.feature import local_binary_pattern
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import uvicorn ## ASGI
from fastapi import FastAPI, File, UploadFile
import tempfile
import cv2
import glob
import os

from mangum import Mangum
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy


svc = joblib.load('ALL.joblib')

#create the app project
app = FastAPI()
handler=Mangum(app)


# store locally
input_folder = tempfile.mkdtemp()

def feature_extractor1(dataset):
    image_dataset = pd.DataFrame()

    
    df = pd.DataFrame()  
    
    img = dataset

    GLCM = greycomatrix(img, [1], [0])      
    GLCM_Energy = greycoprops(GLCM, 'energy')[0]
    df['Energy'] = GLCM_Energy
    GLCM_corr = greycoprops(GLCM, 'correlation')[0]
    df['Corr'] = GLCM_corr       
    GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
    df['Diss_sim'] = GLCM_diss       
    GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
    df['Homogen'] = GLCM_hom       
    GLCM_contr = greycoprops(GLCM, 'contrast')[0]
    df['Contrast'] = GLCM_contr


    GLCM2 = greycomatrix(img, [3], [0])       
    GLCM_Energy2 = greycoprops(GLCM2, 'energy')[0]
    df['Energy2'] = GLCM_Energy2
    GLCM_corr2 = greycoprops(GLCM2, 'correlation')[0]
    df['Corr2'] = GLCM_corr2       
    GLCM_diss2 = greycoprops(GLCM2, 'dissimilarity')[0]
    df['Diss_sim2'] = GLCM_diss2       
    GLCM_hom2 = greycoprops(GLCM2, 'homogeneity')[0]
    df['Homogen2'] = GLCM_hom2       
    GLCM_contr2 = greycoprops(GLCM2, 'contrast')[0]
    df['Contrast2'] = GLCM_contr2

    GLCM3 = greycomatrix(img, [5], [0])       
    GLCM_Energy3 = greycoprops(GLCM3, 'energy')[0]
    df['Energy3'] = GLCM_Energy3
    GLCM_corr3 = greycoprops(GLCM3, 'correlation')[0]
    df['Corr3'] = GLCM_corr3       
    GLCM_diss3 = greycoprops(GLCM3, 'dissimilarity')[0]
    df['Diss_sim3'] = GLCM_diss3       
    GLCM_hom3 = greycoprops(GLCM3, 'homogeneity')[0]
    df['Homogen3'] = GLCM_hom3       
    GLCM_contr3 = greycoprops(GLCM3, 'contrast')[0]
    df['Contrast3'] = GLCM_contr3


    
    # Add more filters as needed
    entropy = shannon_entropy(img)
    df['Entropy'] = entropy

    
    #Append features from current image to the dataset
    image_dataset = image_dataset.append(df)
        
    return image_dataset



def compute_lbp(arr,img):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    # LBP function params
    radius = 3
    n_points = 8 * radius
    n_bins = n_points + 2
    lbp = local_binary_pattern(arr, n_points, radius, 'uniform')
    lbp = lbp.ravel()
    # feature_len = int(lbp.max() + 1)
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)

     # Load the image
    img = cv2.imread(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     # Apply edge detection using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Calculate the contour area
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)

    # Calculate the total area of the image
    total_area = img.shape[0] * img.shape[1]

    # Calculate the ratio of the contour area to the total area
    ratio = area / total_area
    feature=np.append(feature,ratio)
    return feature

#Index route opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {'message': 'Welcome to Spoof Detection'}

# route with a single parameter returns the parameter within the message located at http://127.0.0.1:8000/{Anynamehere}

@app.post('/spoof/')
async def upload_file(file : UploadFile):
    contents = await file.read()
    filename = file.filename
    file_bytes = np.asarray(bytearray(contents), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    cv2.imwrite(input_folder + '/' + filename, opencv_image)
    probability= 0
    print(probability)
    for i, file in enumerate(glob.glob(input_folder +"/*.*")):
        if any([ext in file.lower() for ext in ['.jpeg', '.jpg', '.png']]):
            img=imread(file,as_gray=True)
            img_resize=resize(img,(512,384))
            l=compute_lbp(img_resize,file)
            img_array=cv2.imread(file,0) #this time we will be reading our data in grayscale
            img_resized=resize(img_array,(512,384))
            img_resized = np.uint8(img_resized * 255)
            images=np.array(img_resized,dtype='uint8')
            data = feature_extractor1(images)
            l=np.append(l,data.values.flatten())
            L=l.reshape(1, -1)
            probability=svc.predict_proba(L)
            os.remove(input_folder + '/' + filename)
            return {'file': f'{filename}', 'probability': f'{probability}'}
    
    print(os.listdir(input_folder))
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)