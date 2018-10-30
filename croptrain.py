import json
import os
import os.path as osp
import cv2

traintrain_root = '/mnt/SSD/jzwang/wider/images'
test_root = '/mnt/SSD/jzwang/wider/images'
save_root = '/mnt/SSD/jzwang'

def load_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def check_path(path):
    if not osp.exists(path):
        os.makedirs(path)
        print('path not exist, mkdir:', path)

if __name__ == '__main__':
    train = load_json(osp.join(traintrain_root,'..','label','train.json'))
    test = load_json(osp.join(test_root, '..','label','test.json'))

    train_num, test_num = len(train.keys()), len(test.keys())
    train_cnt, test_cnt = 0, 0
    chk_train, chk_test = False, False
    for movie, info in train.items():
        train_cnt += 1
        candidates = info['candidates']
        candi_len = len(candidates)
        for i, candidate in enumerate(candidates):
            print('train: %d/%d, test: %d/%d ... %s %d/%d'%(train_cnt, train_num, test_cnt, test_num, candidate['img'], i+1, candi_len))
            pid = candidate['id']
            c_label = candidate['label']
            #save_name = '%s.jpg'%pid
            save_name = '%s_%s.jpg'%(c_label, pid)
            img_path = osp.join(traintrain_root, 'train', candidate['img'])
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            bbox = candidate['bbox']
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[0] + bbox[2] > w:
                bbox[2] = w - bbox[0]
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[1] + bbox[3] > h:
                bbox[3] = h - bbox[1]
            crop = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            crop = cv2.resize(crop, (128, 256))
            if chk_train == False:
                check_path(osp.join(save_root, 'wider_exfeat_new', 'train'))
                chk_train = True
            cv2.imwrite(osp.join(save_root, 'wider_exfeat_new/train', save_name), crop)
