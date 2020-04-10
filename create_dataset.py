import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle


path = '/home/joseph/Documents/school/SkinCancer/Data'

#Load Pandas dataset for the labels. 
metadata_df = pd.read_csv('/home/joseph/Documents/school/SkinCancer/HAM10000_metadata.csv')

ids = metadata_df['image_id']
labels = {}

cat_data = {
        'vasc' : 0,
    	'df' : 1,
        'nv' : 2,	
        'mel' : 3,	
        'bkl' : 4,	
        'bcc' : 5,	
        'akiec' : 6,	
}

_c = 0
for _id in ids:
    if _id in labels:
        print('oof....' + _id)
    else:
        _x = metadata_df.loc[metadata_df['image_id'] == _id]['dx']
        # print('-x', _x)
        _tempCat = str(_x.iloc[0])
        # print('_tempCat: ', _tempCat)
        labels[_id] = cat_data[_tempCat]
        # print(_id)
        # print(labels[_id])
        # _c += 1

        # if _c % 35 == 0:
        #     break




print("done...............")
# 192, 256
# size1 = 320 
# size2 = 240
size1 = 256 
size2 = 192
X = []
y = []
n = []
c = 0
for img in os.listdir(path): 
    _img_arr =cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2RGB)
    _img_arr = cv2.cvtColor(_img_arr, cv2.COLOR_BGR2RGB)

    new_arr = cv2.resize(_img_arr, (size1, size2))
    # plt.imshow(new_arr)
    # plt.show()
    # break
    
    _lab = img[:len(img)-4]
    try:
    	n.append(_lab)
    	y.append(labels[_lab])
    	X.append(new_arr)
    except:
    	print(img)

    c+=1
    if (c % 1000 == 0):
    	print(c)


X = np.array(X)
y = np.array(y)
n = np.array(n)


print(X.shape)
print(y.shape)
print(n.shape)
# for i in range(23, 45):
#     print(y[i])
#     print(n[i])
#     # X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)
#     plt.imshow(X[i])
#     plt.show()





print('X pickle time.')
pickle_out = open("Xsmall.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

print('n pickle time.')
pickle_out = open("nsmall.pickle", "wb")
pickle.dump(n, pickle_out)
pickle_out.close()

print('y pickle time.')
pickle_out = open("ysmall.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()