# -*- coding: utf-8 -*- Code from S3Rec repo: https://github.com/aHuiWang/CIKM2020-S3Rec
# @Time    : 2020/4/4 8:18
# @Author  : Hui Wang

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm
import os


def parse(path): # for Amazon
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    # older Amazon
    data_flie = os.path.join(root_path, "raw_datasets", 'reviews_' + dataset_name + '.json.gz')
    # latest Amazon
    for inter in parse(data_flie):
        if float(inter['overall']) <= rating_score: # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        datas.append((user, item, int(time)))
    return datas


def Amazon_meta(dataset_name, data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_flie = os.path.join(root_path, "raw_datasets", 'meta_' + dataset_name + '.json.gz')
    item_asins = list(data_maps['item2id'].keys())
    for info in parse(meta_flie):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas


def Yelp(date_min, date_max, rating_score):
    datas = []
    data_flie = os.path.join(root_path, "raw_datasets", "yelp_kaggle_v2", 'yelp_academic_dataset_review.json')
    lines = open(data_flie).readlines()
    for line in tqdm.tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review['date']
        # 剔除一些例子
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-','').replace(':','').replace(' ','')
        datas.append((user, item, int(time)))
    return datas


def Yelp_meta(datamaps):
    meta_infos = {}
    meta_file = os.path.join(root_path, "raw_datasets", "yelp_kaggle_v2", 'yelp_academic_dataset_business.json')
    item_ids = list(datamaps['item2id'].keys())
    lines = open(meta_file).readlines()
    for line in tqdm.tqdm(lines):
        info = json.loads(line)
        if info['business_id'] not in item_ids:
            continue
        meta_infos[info['business_id']] = info
    return meta_infos


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

# categories 和 brand is all attribute
def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]: # 把主类删除 没有用
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_attribute_Yelp(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] +=1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',') ]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # 对各个数据集得单独排序
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, user_id-1, item_id-1, data_maps


def _is_empty_meta_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _normalize_meta_value(value):
    return "Unknown" if _is_empty_meta_value(value) else value


def _build_selected_meta(info, data_type):
    extracted_meta = {}
    if data_type == 'Amazon':
        extracted_meta['title'] = _normalize_meta_value(info.get('title'))

        price = info.get('price')
        extracted_meta['price'] = str(price) if not _is_empty_meta_value(price) else 'Unknown'

        description = info.get('description')
        extracted_meta['description'] = str(description) if not _is_empty_meta_value(description) else 'Unknown'

        cates = info.get('categories', [])
        if isinstance(cates, list) and cates:
            flat_cates = []
            for sublist in cates:
                if isinstance(sublist, list):
                    flat_cates.extend(sublist)
                else:
                    flat_cates.append(str(sublist))
            extracted_meta['categories'] = ', '.join(flat_cates) if flat_cates else 'Unknown'
        else:
            extracted_meta['categories'] = 'Unknown'

    elif data_type == 'Yelp':
        extracted_meta['name'] = str(_normalize_meta_value(info.get('name')))

        address = info.get('address')
        extracted_meta['address'] = str(address) if not _is_empty_meta_value(address) else 'Unknown'

        city = info.get('city')
        extracted_meta['city'] = str(city) if not _is_empty_meta_value(city) else 'Unknown'

        state = info.get('state')
        extracted_meta['state'] = str(state) if not _is_empty_meta_value(state) else 'Unknown'

        postal_code = info.get('postal_code')
        extracted_meta['postal_code'] = str(postal_code) if not _is_empty_meta_value(postal_code) else 'Unknown'

        cates = info.get('categories')
        extracted_meta['categories'] = str(cates) if not _is_empty_meta_value(cates) else 'Unknown'

    return extracted_meta


def build_id2meta_collections(data_maps, meta_infos, data_type):
    id2meta = {}

    for processed_id, raw_id in data_maps['id2item'].items():
        if raw_id not in meta_infos:
            raise ValueError(f"Raw item ID {raw_id} not found in meta_infos for processed ID {processed_id}")

        info = meta_infos[raw_id]
        id2meta[processed_id] = _build_selected_meta(info, data_type)

    return id2meta


def main(data_name, data_type='Amazon'):
    assert data_type in {'Amazon', 'Yelp'}
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    else:
        datas = Amazon(data_name+'_5', rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_name} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)


    print('Begin extracting meta infos...')

    if data_type == 'Amazon':
        meta_infos = Amazon_meta(data_name, data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Amazon(meta_infos, data_maps, attribute_core)
    else:
        meta_infos = Yelp_meta(data_maps)
        attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_Yelp(meta_infos, data_maps, attribute_core)

    print(f'{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')
    
    id2meta = build_id2meta_collections(data_maps, meta_infos, data_type)

    # -------------- Save Data ---------------
    processed_data_path = os.path.join(root_path, "datasets", "sequential", data_name)
    os.makedirs(processed_data_path, exist_ok=True)
    data_file = os.path.join(processed_data_path, data_name + '.txt')
    item2attributes_file = os.path.join(processed_data_path, data_name + '_item2attributes.json')
    id2meta_file = os.path.join(processed_data_path, data_name + '_id2meta.json')

    with open(data_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')
    json_str = json.dumps(item2attributes)
    with open(item2attributes_file, 'w') as out:
        out.write(json_str)

    with open(id2meta_file, 'w') as out:
        json.dump(id2meta, out, indent=4)
        
    # Save reverse mappings for future reverse engineering
    mapping_file = os.path.join(processed_data_path, data_name + '_mappings.json')
    with open(mapping_file, 'w') as out:
        json.dump({
            'id2item': datamaps['id2item'],
            'id2attribute': datamaps['id2attribute']
        }, out)


if __name__ == '__main__':
    root_path = "../"
    # amazon data is the 2014 version, download 5 cores and metadata https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
    amazon_datas = ['Toys_and_Games', 'Beauty']

    for name in amazon_datas:
        main(name, data_type='Amazon')
    
    # yelp dataset from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/2
    main('Yelp', data_type='Yelp')
