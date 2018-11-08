import json
import numpy as np
import argparse
import os.path as osp

from eval import eval
from utils.pkl import my_unpickle
from scipy.spatial.distance import pdist, squareform

def load_json(name):
    with open(name) as f:
        data = json.load(f)
        return data

def read_info(info_file):
    with open(info_file) as f:
        info_dict = json.load(f)
    return info_dict

def read_feat(feat_file):
    with open(feat_file) as f:
        feat_list = json.load(f)
    feat_dict = {}
    for x in feat_list:
        feat_dict[x['id']] = x
    return feat_dict


def load_face(val_data, face_data):
    face_dict = {}
    movie_list = []
    for movie, info in val_data.items():
        movie_list.append(movie)
        casts = info['cast']
        candidates = info['candidates']
        cast_ids, cast_ffeats = [], []
        cast_ids = [x['id'] for x in casts]
        for key in cast_ids:
            #cast_ids.append(key['id'])
            cast_ffeats.append(face_data[key]['feat'])
        cast_ffeats = np.array(cast_ffeats)
        #print('cast_ffeats.shape:',cast_ffeats.shape)
        candi_f_ids, candi_f_ffeats = [], []
        candi_f_ids_old = [x['id'] for x in candidates]
        for key in candi_f_ids_old:
       	    tmp = face_data[key]['feat']
            if tmp is not None:
                feat = np.array(tmp)
                candi_f_ffeats.append(feat)
                candi_f_ids.append(face_data[key]['id'])
            #else:
            #    candi_f_ids.remove(key)
        candi_f_ffeats = np.array(candi_f_ffeats)
        #print(candi_f_ffeats)
        #print("candi_f_ffeats.shape:",candi_f_ffeats.shape)
        face_dict.update(
            {
                movie:{
                    'cast_ids':cast_ids,
                    'cast_ffeats':cast_ffeats,
                    'candi_f_ids': candi_f_ids,
                    'candi_f_ffeats':candi_f_ffeats,
                }
            }
        )
    return face_dict, movie_list

def rank(movie_face, movie_reid):
    cast_ids, cast_ffeats = movie_face['cast_ids'], movie_face['cast_ffeats']
    candi_f_ids, candi_f_ffeats = movie_face['candi_f_ids'], movie_face['candi_f_ffeats']
    candi_ids, candi_feats = movie_reid['candi_ids'], movie_reid['candi_feats']
    movie_rank = {cast_id:[] for cast_id in cast_ids}

    cast_candi_fsim = np.dot(cast_ffeats, candi_f_ffeats.T)
    candi_candi_fsim = np.dot(candi_f_ffeats, candi_f_ffeats.T)
    # print(candi_feats.shape)
    candi_candi_dist = pdist(candi_feats, 'euclidean')
    candi_candi_dist = squareform(candi_candi_dist)
    #print("cast_candi_fsim.shape[0],len(cast_ids),cast_candi_fsim.shape[1],len(candi_f_ids):",cast_candi_fsim.shape[0],len(cast_ids),cast_candi_fsim.shape[1],len(candi_f_ids))


def rank2txt(rank, file_name):
    with open(file_name, 'w') as f:
        for cast_id, candi_ids in rank.items():
            line = '%s %s\n'%(cast_id, ','.join(candi_ids))
            f.write(line)

def rank_eval(res, label):
    all_ap = eval(res, label)
    return np.array(all_ap)

def main(args):
    #face_pkl = my_unpickle(osp.join('./features', face_feat_name))
    #face_dict, movie_list = load_face(face_pkl)
    info_file = '/mnt/SSD/jzwang/wider/label/val.json'
    info_dict = read_info(info_file)
    feat_file = '/home/jzwang/new/baseline/FaceTool/face_val.json'
    feat_dict = read_feat(feat_file)
    face_dict, movie_list = load_face(info_dict, feat_dict)
    print('Done !')

    rank_list = {}
    movie_num = len(movie_list)
    for i, movie in enumerate(movie_list):
        movie_face = face_dict[movie]
        movie_reid = reid_dict[movie]
        movie_rank, recall_num = rank(movie_face, movie_reid)
        print('movie: %s, %d/%d, recall num: %d'%(movie, i+1, movie_num, recall_num))
        rank_list.update(movie_rank)

    if args.is_test == '1':
        rank2txt(rank_list, 'test_rank.txt')
    else:
        rank2txt(rank_list, 'val_rank.txt')
        all_ap = rank_eval('val_rank.txt', 'val_label.json')
        print(np.sort(all_ap)[::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-test', type=str, default='0', choices=['0', '1'])
    args = parser.parse_args()
    main(args)
