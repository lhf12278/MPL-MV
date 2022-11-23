import cv2
import os
import shutil
from tqdm import tqdm
def swap_channel(img_path):
    img = cv2.imread(img_path)

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # cv2.imshow("Red", R)
    # cv2.imshow("Green", G)
    # cv2.imshow("Blue", B)

    merged1 = cv2.merge([G, R, B])
    merged2 = cv2.merge([G, B, R])
    merged3 = cv2.merge([B, R, G])
    merged4 = cv2.merge([R, G, B])
    merged5 = cv2.merge([R, B, G])
    merged6 = cv2.merge([B, G, R])
    img_list = [merged1,merged2,merged3,merged4,merged5,merged6]
    return img_list
    # cv2.imshow('1',img)
    # cv2.waitKey(0)



if __name__ == '__main__':
    img_path = 'E:/reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_train/'
    out_path = 'E:/reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_train2/'
    img_list = os.listdir(img_path)
    for i  in tqdm(range(len(img_list))):
        img = img_list[i]
        pid = i
        c_id = img.split('_')[1]
        jpg = img.split('_')[2]
        a_img_path = os.path.join(img_path,img)
        channel_img_list = swap_channel(a_img_path)
        for j in range(len(channel_img_list)):
            channel_img = channel_img_list[j]
            channel_img_name = str(pid).zfill(5)+'_'+str(c_id)+'_'+str(j)+jpg
            channel_img_path = os.path.join(out_path,channel_img_name)
            cv2.imwrite(channel_img_path,channel_img)



