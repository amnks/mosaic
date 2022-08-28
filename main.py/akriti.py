import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Activation, Dropout, Flatten, Dense, BatchNormalization



def predict(image):
    rgb_planes = cv2.split(image)
    r,g,b = rgb_planes
    # cv2.imshow('red', r)
    # cv2.imshow('green', g)
    # cv2.imshow('blue', b)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
         dilated_img = cv2.dilate(plane, np.ones((15,15), np.uint8))
         bg_img = cv2.medianBlur(dilated_img, 21)
         diff_img = 255 - cv2.absdiff(plane, bg_img)
         norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
         result_planes.append(diff_img)
         result_norm_planes.append(norm_img)

    result1 = cv2.merge(result_planes)
    import matplotlib.pyplot as plt
    result_norm = cv2.merge(result_norm_planes)

    #plt.imshow(result_norm)
    #cv2.imshow('shadows_out.png', result1)
    #cv2.imshow('shadows_out_norm.png', result_norm)

        # cv2.waitKey(0)
    # blur = cv2.GaussianBlur(img_erosion,(5,5),2)
    # plt.imshow(blur)
    kernel = np.ones((1,1), np.uint8)                  #yaha 10 ki jagah 5 kiye
    gray= cv2.cvtColor(result_norm,cv2.COLOR_BGR2GRAY) #yaha ham result 1 ki jagah norm likh diye
    #cv2.imshow('gray', gray)
    img_erosion = cv2.erode(gray, kernel, iterations=1) 
    #cv2.imshow('img_erosion', img_erosion)
    original_image= result1


    edges= cv2.Canny(gray, 50,200)
    #cv2.imshow('egeg', edges)

    # ret,thresh1 = cv2.threshold(img_erosion,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # #cv2.imshow("kme",thresh1)
    # contours, hierarchy= cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(thresh1,(17,17),0)
    thresh1 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
       
    #cv2.imshow("thresh1",thresh1)
    blur = cv2.GaussianBlur(thresh1,(17,17),0)
    thresh2 = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    #kernel = np.ones((1,1), np.uint8)
    #thresh2 = cv2.erode(thresh2, kernel, iterations=5) 
    #cv2.imshow("thresh2",thresh2)

    contours, hierarchy= cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) != 0:
        # draw in blue the contours that were founded
        # cv2.drawContours(image, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    #     cv.rectangle(result1,(x,y),(x+w,y+h),(255,255),2)
    #     rect = cv.minAreaRect(c)
    #     box = cv.boxPoints(rect)
    #     box = np.int0(box)
    #     cv.drawContours(image,[box],0,(0,0,255),2)


        # draw the biggest contour (c) in green
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
   

        #plt.imshow( result2)
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    ar=h*w
    #cropimg = thresh1[y:y+h,x:x+w]
    #cropimggray = gray[y:y+h,x:x+w]
    #plt.imshow(cropimggray)
    rows,cols=thresh2.shape
    for i in range(rows):
      for j in range(cols):
        thresh2[i][j]/=255


    row_sums = thresh2.sum(axis=1)/cols

        
    for i in range(rows):
        for j in range(cols):
            thresh2[i][j]*=255
            
    # print(row_sums)
    # print(type(row_sums))
    ind=-1
    flag=0
    for i in row_sums:
        ind+=1
        #print(i,ind)
        if i>0.2:
            flag=1
            for j in range(cols):
                for k in range(30):
                    thresh2[ind+k][j]=0
                for k in range(10):
                    thresh2[ind-k][j]=0
            if flag==1:
                flag=2
        if flag==2:
            break
        
    no_danda=thresh2
    cv2.imshow('no_danda', no_danda)
    
    kernel = np.ones((2,2), np.uint8)  #10 ki jagah 8 kiye hai
    img_min = cv2.erode(no_danda, kernel, iterations=8)
    #cv2.imshow("img_minb",img_min)
    #plt.imshow(img_min)
    blackhat = cv2.morphologyEx(img_min, cv2.MORPH_TOPHAT, kernel)
    #cv2.imshow("black",blackhat)
    contours2, hierarchy2= cv2.findContours(no_danda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # imgi = cv2.drawContours(thresh2, contours2, -1, (0,0,255), 3)
    ROI_number = 0
    # cv2.imshow('Larged', imgi)
    y=0
    h=0
    k=0
    lst=[]
    wdth=[]
    for ct in contours2:
        if cv2.contourArea(ct)>0.005*ar:
          x,y,w,h = cv2.boundingRect(ct)
        # cv2.rectangle(upper,(x,y),(x+w,y+h),(255,25,70),2)
          wdth.append([x,x+w])
    

    wdth=sorted(wdth)
    for i in wdth:
        ROI = gray[y-35:y+h+10,i[0]-2:i[1]+2]
        ret2,thresh4 = cv2.threshold(ROI,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(thresh4,(5,5),1)
        #ret,thresh5  = cv2.threshold(ROI,0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        #close = cv2.morphologyEx(thresh5, cv2.MORPH_CLOSE, kernel, iterations=1)
        outputImage = cv2.copyMakeBorder(
                      blur, 
                      2, 
                      2, 
                      2, 
                      2, 
                      cv2.BORDER_CONSTANT, 
                      value=(0,0)
                   )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
        outputImage = cv2.morphologyEx(outputImage, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        #kernel=np.ones((4,4), np.uint8)
        #xyz=cv2.erode(outputImage,kernel,iterations=1)
        #plt.imshow(blur)
        cv2.imwrite(str(k) + '.jpeg',outputImage)

        lst.append(str(k)+'.jpeg')
        k +=1
        # a = cropimggray[:,x:x+w]

        # cv2.drawContours(image, contours, -1, 255, 3)








    return lst


    if (cv2.waitKey(0) & 0xFF == 27):
        cv2.destroyAllWindows()


def test(lst):
    import cv2
    from keras.preprocessing.image import img_to_array
    from keras.models import load_model
    import numpy as np
    import argparse
    #import imutils
    from keras.preprocessing import image
    import numpy as np
    from keras.preprocessing import image
    from matplotlib import pyplot as plt
    import tensorflow as tf
    #plt.imshow(test_image, interpolation='nearest')
    #plt.show()
    model=tf.keras.models.load_model("train_model.h5")
    final=[]
    cnt=0
    for i in range(len(lst)):
        test_image = cv2.imread(lst[i])
        image = cv2.resize(test_image, (32,32))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
    #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    #print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #plt.imshow(image, interpolation='nearest')
        #plt.show()
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=3)
        #print("[INFO] loading network...")

        labels = ['ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'क', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'ख', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ','०' , '१', '२', '३', '४', '५', '६', '७', '८', '९']
        lists = model.predict(image)[0]
        #print("The letter is ",labels[np.argmax(lists)])
        final+=(labels[np.argmax(lists)])
        cnt+=1

    #print(cnt)
    return final


if __name__=="__main__":
    image=cv2.imread("kamansav.jpeg")
    lst_final=predict(image)
    ans=test(lst_final)
    print(ans)

