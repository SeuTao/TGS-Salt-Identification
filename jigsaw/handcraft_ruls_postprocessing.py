from include import *

train_mask_path = os.path.join(root_dir, r'train/masks')
train_data_dir = os.path.join(root_dir, r'train/images')
test_data_dir = os.path.join(root_dir, r'test/images')

vertical_y_ratio = 0.8
# build train img set, test img set
train_set = set()
test_set = set()
for subdir in os.listdir(train_data_dir):
    train_set.add(subdir.split('.')[0])
for subdir in os.listdir(test_data_dir):
    test_set.add(subdir.split('.')[0])

def create_submission(predictions):
    output = []
    for image_id, mask in predictions:
        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([image_id, rle_encoded])
    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    return submission

def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]
    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b
    if len(rle) != 0 and rle[-1] + rle[-2] > (x.size+1):
        rle[-2] = rle[-2] - 1
    return rle

# detect semi vertical mask
def detect_semi_vertical_mask(img):
    # semi vertical threshlod
    h_offset = int(101 * vertical_y_ratio)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(img[100]) == 255 or np.mean(img[100]) == 0:
        is_semi_vertical = False
        return is_semi_vertical
    is_semi_vertical = np.all(img[h_offset:] == img[100])
    if np.all(img == img[100]):
        is_semi_vertical = False
    return is_semi_vertical

# detect full vertical mask
def detect_full_vertical_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if np.mean(img[100]) == 255 or np.mean(img[100]) == 0:
        is_full_vertical = False
        return is_full_vertical
    is_full_vertical = np.all(img == img[100])
    return is_full_vertical

################  load train mask ##################
def load_train_mask(train_mask_path):
    train_mask = {}
    for subdir in os.listdir(train_mask_path):
        imgpath = os.path.join(train_mask_path, subdir)
        mask = cv2.imread(imgpath)
        train_mask.setdefault(subdir.split('.')[0], mask)
    return train_mask

##############  make test mask dict  ###############
def do_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H,W), np.uint8)
    if type(rle).__name__ == 'float': return mask
    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

def decode_csv(csv_name):
    import pandas as pd
    data = pd.read_csv(csv_name)
    id = data['id']
    rle_mask = data['rle_mask']
    dict = {}
    for id, rle in zip(id,rle_mask):
        tmp = do_length_decode(rle, 101, 101, fill_value=1)
        tmp *= 255
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        dict[id] = tmp
    return dict

def load_test_mask(test_mask_path):
    mask_test = decode_csv(test_mask_path)
    return mask_test

#############  find all full vertical mask ##############
def find_all_vertical_mask():
    train_vertical_dict = {}
    train_semi_vertical_dict = {}
    train_mask = load_train_mask(train_mask_path)
    for k, v in train_mask.items():
        is_semi_v = detect_semi_vertical_mask(v)
        if is_semi_v:
            train_semi_vertical_dict.setdefault(k, 'semi')
        is_full_v = detect_full_vertical_mask(v)
        if is_full_v:
            train_vertical_dict.setdefault(k, 'full')
    with open(r'./jigsaw/json_files/train_vertical_mask.json', 'w') as f:
        json.dump(train_vertical_dict, f)
    with open(r'./jigsaw/json_files/train_semi_vertical_mask.json', 'w') as f:
        json.dump(train_semi_vertical_dict, f)

#################### vertical rules ####################
def vertical_rule1(img_map, map_img, vertical_mask):
    test_vertical_mask = {}
    for k, v in vertical_mask.items():
        if k in img_map.keys():
            mapid, x, y = img_map[k]['mapid'], img_map[k]['x'], img_map[k]['y']
            try:
                up_imgid = map_img[mapid][x][str(int(y) - 1)]
            except:
                up_imgid = 'None'
            if up_imgid in test_set:
                if up_imgid not in test_vertical_mask.keys():
                    test_vertical_mask.setdefault(up_imgid, k)

    with open(r'./jigsaw/json_files/vertical_test_rule1.json', 'w') as f:
        json.dump(test_vertical_mask, f)
    print('Rule1 detect test vertical : ', len(test_vertical_mask.keys()))

def vertical_rule2(vertical_mask, img_map, map_img):
    test_vertical_mask = {}

    masks = vertical_mask
    json_file = r'./jigsaw/json_files/vertical_test_rule2.json'

    for k, v in masks.items():
        if k in img_map.keys():
            mapid, x, y = img_map[k]['mapid'], img_map[k]['x'], img_map[k]['y']
            for key, value in map_img[mapid][x].items():
                if value in test_set and (int(key) > int(y)):
                    test_vertical_mask.setdefault(value, k)

    with open(json_file, 'w') as f:
        json.dump(test_vertical_mask, f)
    print('Rule4 detect test vertical : ', len(test_vertical_mask.keys()))

def semi_vertical_rule(semi_vertical_mask, img_map, map_img):
    test_vertical_mask = {}

    masks = semi_vertical_mask
    json_file = r'./jigsaw/json_files/semi_vertical.json'

    for k, v in masks.items():
        if k in img_map.keys():
            mapid, x, y = img_map[k]['mapid'], img_map[k]['x'], img_map[k]['y']
            for key, value in map_img[mapid][x].items():
                if value in test_set and (int(key) > int(y)):
                    test_vertical_mask.setdefault(value, k)

    with open(json_file, 'w') as f:
        json.dump(test_vertical_mask, f)
    print('Rule4 detect test vertical : ', len(test_vertical_mask.keys()))

def vertical_rule3(vertical_mask, test_mask_path, img_map, map_img):
    test_vertical_mask1 = {}
    test_vertical_mask3_2 = {}
    test_vertical_mask4 = {}
    test_vertical_mask5 = {}
    test_mask = load_test_mask(test_mask_path)
    #########################################
    for k, v in vertical_mask.items():
        if k not in img_map.keys(): continue
        mapid, x, y = img_map[k]['mapid'], img_map[k]['x'], img_map[k]['y']
        for key, value in map_img[mapid][x].items():
            if value not in test_set: continue
            # find max train image, min test image, min big test image, in same columns
            max_y, min_y_test, min_y_test_m = -1, 100, 100
            for kk, vv in map_img[mapid][x].items():
                if vv in train_set and (vv != k) and (vv not in vertical_mask.keys()) and (int(kk) < int(y)) and (
                        int(kk) > max_y):
                    max_y = int(kk)
                if vv in test_set:
                    mask_test = test_mask[vv]
                    if np.sum(mask_test) != 0 and (int(kk) < min_y_test):
                        min_y_test = int(kk)
                    if np.sum(mask_test) > 50 * 50 * 255 and (int(kk) < min_y_test_m):
                        min_y_test_m = int(kk)

            # apply rule5-1, 5-5
            if int(key) < int(y) and (int(key) > max_y and max_y != -1):
                if map_img[mapid][x][str(max_y)] in train_set:
                    imgpath = os.path.join(train_mask_path, map_img[mapid][x][str(max_y)] + '.png')
                    mask = cv2.imread(imgpath)
                    # rule5-1
                    if np.sum(mask) == 0:
                        test_vertical_mask1.setdefault(value, k)
                    # rule5-5
                    else:
                        test_vertical_mask5.setdefault(value, k)

            # apply rule5-3-2
            if int(key) < int(y) and (int(key) > max_y and max_y == -1) and min_y_test_m != 100 and (
                    int(key) > min_y_test_m):
                test_vertical_mask3_2.setdefault(value, k)
            if int(key) < int(y) and (int(key) > max_y and max_y == -1) and min_y_test != 100:
                if np.sum(test_mask[map_img[mapid][x][str(min_y_test)]]) <= 50 * 50 * 255 and np.sum(
                        test_mask[value]) <= 10 * 255 and (int(key) > min_y_test):
                    test_vertical_mask3_2.setdefault(value, k)
            # apply rule 5-4
            if int(key) < int(y) and (int(key) > max_y and max_y == -1) and (min_y_test == 100) and (
                    int(key) >= int(int(y) / 2)):
                test_vertical_mask4.setdefault(value, k)

    with open(r'./jigsaw/json_files/vertical_test_rule3_1.json', 'w') as f:
        json.dump(test_vertical_mask1, f)
    with open(r'./jigsaw/json_files/vertical_test_rule3_2.json', 'w') as f:
        json.dump(test_vertical_mask3_2, f)
    with open(r'./jigsaw/json_files/vertical_test_rule3_3.json', 'w') as f:
        json.dump(test_vertical_mask4, f)
    with open(r'./jigsaw/json_files/vertical_test_rule3_4.json', 'w') as f:
        json.dump(test_vertical_mask5, f)

    print('test_vertical_mask1_num : '+str( len(test_vertical_mask1.keys())))
    print('test_vertical_mask3_num : '+str( len(test_vertical_mask3_2.keys())))
    print('test_vertical_mask4_num : '+str( len(test_vertical_mask4.keys())))
    print('test_vertical_mask5_num : '+str( len(test_vertical_mask5.keys())))

def vertical_rules_combine(rule):
    vertical_test_rule = {}
    rule_name = ''
    for i in rule:
        json_file_path = r'./jigsaw/json_files/vertical_test_rule' + i + '.json'
        print(json_file_path)

        rule_name += '-'+ i
        with open(json_file_path, 'r') as f:
            vertical_test_rule_tmp = json.load(f)
            for k, v in vertical_test_rule_tmp.items():
                if k in vertical_test_rule.keys(): continue
                vertical_test_rule.setdefault(k, v)

        os.remove(json_file_path)

    print('Vertical test rule number : ', len(vertical_test_rule.keys()))
    with open(r'./jigsaw/json_files/vertical.json', 'w') as f:
        json.dump(vertical_test_rule, f)

################## empty rules ###################
def build_full_train_mask(train_mask_path, img_map, img_map_set):
    mask = img_map.copy()
    train_mask = load_train_mask(train_mask_path)
    train_full_set = train_set.copy()
    for i in (train_set & img_map_set):
        train_mask_tmp = cv2.cvtColor(train_mask[i], cv2.COLOR_BGR2GRAY)
        if np.sum(train_mask_tmp) == 0: continue
        mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
        # left
        if np.sum(train_mask_tmp[:, 0]) > 255:  # 0~
            mask.setdefault('L-' + i, {'x': str(int(x) - 1), 'y': y, 'mapid': mapid})
            train_full_set.add('L-' + i)
        elif np.mean(train_mask_tmp[100]) == 255:
            mask.setdefault('L-' + i, {'x': str(int(x) - 1), 'y': str(int(y) + 1), 'mapid': mapid})
            train_full_set.add('L-' + i)

        # right
        if np.sum(train_mask_tmp[:, 100]) > 255:  # 0~
            mask.setdefault('R-' + i, {'x': str(int(x) + 1), 'y': y, 'mapid': mapid})
            train_full_set.add('R-' + i)
        elif np.mean(train_mask_tmp[100]) == 255:
            mask.setdefault('R-' + i, {'x': str(int(x) + 1), 'y': str(int(y) + 1), 'mapid': mapid})
            train_full_set.add('R-' + i)

    with open(r'./jigsaw/json_files/jigsaw_map_virtual.json', 'w') as f:
        json.dump(mask, f)

def empty_rule1(train_mask_path, img_map, img_map_set, map_img):
    mask = {}
    train_mask = load_train_mask(train_mask_path)
    for i in (train_set & img_map_set):
        if np.mean(train_mask[i][100, :, :]) == 255:
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            for y_, imgid in map_img[mapid][x].items():
                if imgid in test_set and (int(y_) > int(y)) and int(mapid) <= 158: mask.setdefault(imgid, 'Empty')

    with open(r'./jigsaw/json_files/empty_rule1.json', 'w') as f:
        json.dump(mask, f)

def empty_rule2(train_mask_path, test_mask_path, img_map, img_map_set, map_img, map_img_virtual):
    mask = {}
    train_mask = load_train_mask(train_mask_path)
    test_mask = load_test_mask(test_mask_path)
    for i in (train_set & img_map_set):
        if np.sum(train_mask[i]) == 0:
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            flag = 0
            for y_, imgid in map_img[mapid][x].items():
                if imgid in train_set: img_mask = train_mask[imgid]
                else: img_mask = test_mask[imgid]
                if (int(y_) < int(y)) and np.sum(img_mask) != 0 and int(mapid) <= 158: flag = 1
            for y_idx in range(-1, int(y)):
                if mapid in map_img_virtual.keys() and x in map_img_virtual[mapid].keys() and str(y_idx) in map_img_virtual[mapid][x].keys() and \
                        ('-' in  map_img_virtual[mapid][x][str(y_idx)]): flag = 1
            if flag == 0: continue
            for y_, imgid in map_img[mapid][x].items():
                if imgid in test_set and (int(y_) > int(y)) and int(mapid) <= 158: mask.setdefault(imgid, 'Empty')

    with open(r'./jigsaw/json_files/empty_rule2.json', 'w') as f:
        json.dump(mask, f)

def make_sub_rule_empty(test_mask_path):
    empty_mask = {}
    test_mask = load_test_mask(test_mask_path)
    output_test_mask_path = test_mask_path.replace('.csv', '') + '-empty.csv'
    out = []
    with open(r'./jigsaw/json_files/empty_rule1.json') as f:
        rule1_dict = json.load(f)
        empty_mask.update(rule1_dict)
    with open(r'./jigsaw/json_files/empty_rule2.json') as f:
        rule2_dict = json.load(f)
        empty_mask.update(rule2_dict)

    for id in test_mask:
        sum_tmp = test_mask[id][:, :, 0] / 255
        if id in empty_mask.keys():
            sum_tmp = np.zeros([101, 101]).astype(np.uint8)
        out.append([id, sum_tmp])
    submission = create_submission(out)
    submission.to_csv(output_test_mask_path, index=None)

    os.remove(r'./jigsaw/json_files/empty_rule1.json')
    os.remove(r'./jigsaw/json_files/empty_rule2.json')

################## smooth rules #################
def padding(input, output, img_map, map_img, test_mask):
    with open(input) as f:
        rule6_dict = json.load(f)
    rule6_test_dict = rule6_dict.copy()
    for k, v in rule6_dict.items():
        mapid, x, y = img_map[k]['mapid'], img_map[k]['x'], img_map[k]['y']
        if str(int(y) - 1) in map_img[mapid][x].keys():
            up = map_img[mapid][x][str(int(y) - 1)]
            if up in test_set:
                if 'up' not in v.keys() and (np.mean(test_mask[up][100, :]) > 0):
                    rule6_test_dict.setdefault(k, {}).setdefault('up', test_mask[up][100, :, 0].tolist())
        if str(int(y) + 1) in map_img[mapid][x].keys():
            down = map_img[mapid][x][str(int(y) + 1)]
            if down in test_set:
                if 'down' not in v.keys() and (np.mean(test_mask[down][0, :]) > 0):
                    rule6_test_dict.setdefault(k, {}).setdefault('down', test_mask[down][0, :, 0].tolist())
        if str(int(x) - 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) - 1)].keys():
            left = map_img[mapid][str(int(x) - 1)][y]
            if left in test_set:
                if 'left' not in v.keys() and (np.mean(test_mask[left][:, 100]) > 0):
                    rule6_test_dict.setdefault(k, {}).setdefault('left', test_mask[left][:, 100, 0].tolist())
        if str(int(x) + 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) + 1)].keys():
            right = map_img[mapid][str(int(x) + 1)][y]
            if right in test_set:
                if 'right' not in v.keys() and (np.mean(test_mask[right][:, 0]) > 0):
                    rule6_test_dict.setdefault(k, {}).setdefault('right', test_mask[right][:, 0, 0].tolist())
    with open(output, 'w') as f:
        json.dump(rule6_test_dict, f)

def smooth_rule(train_mask_path, test_mask_path, vertical_mask, img_map, img_map_set, map_img):
    mask = {}
    train_mask = load_train_mask(train_mask_path)
    test_mask = load_test_mask(test_mask_path)
    for i in (train_set & img_map_set):
        if i in vertical_mask.keys(): continue
        train_mask_tmp = cv2.cvtColor(train_mask[i], cv2.COLOR_BGR2GRAY)
        if np.sum(train_mask_tmp) == 0: continue
        # up
        if np.mean(train_mask_tmp[0]) < 255 and (np.sum(train_mask_tmp[0]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(y) - 1) in map_img[mapid][x].keys() and int(mapid) < 159:
                imgid = map_img[mapid][x][str(int(y) - 1)]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("down", train_mask_tmp[0].tolist())
        # down
        if np.mean(train_mask_tmp[100]) < 255 and (np.sum(train_mask_tmp[100]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(y) + 1) in map_img[mapid][x].keys() and int(mapid) < 159:
                imgid = map_img[mapid][x][str(int(y) + 1)]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("up", train_mask_tmp[100].tolist())
        # left
        if np.mean(train_mask_tmp[:, 0]) < 255 and (np.sum(train_mask_tmp[:, 0]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(x) - 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) - 1)].keys() and int(mapid) < 159:
                imgid = map_img[mapid][str(int(x) - 1)][y]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("right", train_mask_tmp[:, 0].tolist())
        # right
        if np.mean(train_mask_tmp[:, 100]) < 255 and (np.sum(train_mask_tmp[:, 100]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(x) + 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) + 1)].keys() and int(mapid) < 159:
                imgid = map_img[mapid][str(int(x) + 1)][y]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("left", train_mask_tmp[:, 100].tolist())
    with open(r'./jigsaw/json_files/smooth_rule.json', 'w') as f:
        json.dump(mask, f)
    padding(r'./jigsaw/json_files/smooth_rule.json',
            r'./jigsaw/json_files/smooth_rule_test.json', img_map, map_img, test_mask)

def liner_line(x1, y1, x2, y2):
    if y2 - y1 == 0: return 10000000000
    k = float(x2 - x1) / float(y2 - y1)
    return k

def for_smooth_rule():
    test_mask_dict = {}
    with open(r'./jigsaw/json_files/smooth_rule_test.json') as f:
        rule6_dict = json.load(f)
    for k, v in rule6_dict.items():
        new_mask = np.zeros((101, 101))
        if len(v.keys()) == 1:
            if 'up' in v.keys():
                new_mask += np.array([v['up'] for i in range(101)])
            if 'down' in v.keys():
                new_mask += np.array([v['down'] for i in range(101)])
            if 'left' in v.keys():
                new_mask_tmp = np.array([v['left'] for i in range(101)])
                new_mask_tmp = new_mask_tmp.transpose((1, 0))
                new_mask += new_mask_tmp
            if 'right' in v.keys():
                new_mask_tmp = np.array([v['right'] for i in range(101)])
                new_mask_tmp = new_mask_tmp.transpose((1, 0))
                new_mask += new_mask_tmp
        if len(v.keys()) == 2:
            # up - down
            if 'up' in v.keys() and 'down' in v.keys():
                up_first, up_last = np.where(np.array(v['up'])==255)[0][0], \
                                    np.where(np.array(v['up'])==255)[0][-1]
                down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                        np.where(np.array(v['down']) == 255)[0][-1]
                k1, k2 = liner_line(up_first, 0, down_first, 100), liner_line(up_last, 0, down_last, 100)
                for i in range(new_mask.shape[0]):
                    new_mask[i, max(0, int(k1 * i + up_first)) : min(101, int(k2 * i + up_last))] = 255
            if 'left' in v.keys() and 'right' in v.keys():
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                k1 = liner_line(left_first, 0, right_first, 100)
                for i in range(new_mask.shape[1]):
                    new_mask[min(100, max(0, int(k1 * i + left_first))):, i] = 255

            if 'up' in v.keys() and 'left' in v.keys():
                up_first, up_last = np.where(np.array(v['up']) == 255)[0][0], \
                                    np.where(np.array(v['up']) == 255)[0][-1]
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                if left_first == 0: left_first += 1
                k1 = liner_line(up_first, 0, 0, left_first)
                for i in range(new_mask.shape[0]):
                    new_mask[i, max(0, int(k1 * i + up_first)) : up_last] = 255
            if 'up' in v.keys() and 'right' in v.keys():
                up_first, up_last = np.where(np.array(v['up']) == 255)[0][0], \
                                    np.where(np.array(v['up']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                if right_first == 0: right_first += 1
                k2 = liner_line(up_last, 0, 100, right_first)
                for i in range(new_mask.shape[0]):
                    new_mask[i, up_first: min(101, int(k2 * i + up_last))] = 255
            if 'down' in v.keys() and 'left' in v.keys():
                down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                        np.where(np.array(v['down']) == 255)[0][-1]
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                k1 = liner_line(0, left_first, down_last, 100)
                for i in range(left_first, 101):
                    new_mask[i, 0 : min(101, int(k1 * (i - left_first)))] = 255
            if 'down' in v.keys() and 'right' in v.keys():
                down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                        np.where(np.array(v['down']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                k1 = liner_line(100, right_first, down_first, 100)
                for i in range(right_first, 101):
                    new_mask[i, max(0, int(k1 * (i - right_first) + 100)): 101] = 255
        if len(v.keys()) == 3:
            if 'down' in v.keys() and 'left' in v.keys() and 'right' in v.keys():
                down_first, down_last = np.where(np.array(v['down']) == 0)[0][0], \
                                        np.where(np.array(v['down']) == 0)[0][-1]
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                k1, k2 = liner_line(0, left_first, down_first, 100), liner_line(100, right_first, down_last, 100)
                for i in range(left_first, 101):
                    new_mask[i, 0 : min(101, int(k1 * (i - left_first)))] = 255
                for i in range(right_first, 101):
                    new_mask[i, max(0, int(k2 * (i - right_first) + 100)): 101] = 255
        new_mask = new_mask[:, :, np.newaxis]
        if np.sum(new_mask) != 0:
            test_mask_dict[k] = np.reshape(new_mask, (101, 101))
    print("Smooth Rule1 for train detect : ", len(test_mask_dict.keys()))

    os.remove(r'./jigsaw/json_files/smooth_rule_test.json')
    return test_mask_dict

def smooth_rule_on_test(test_mask_path, vertical_mask, semi_vertical_mask, img_map, img_map_set, map_img):
    mask = {}
    test_mask = load_test_mask(test_mask_path)
    for i in (test_set & img_map_set):
        if i in vertical_mask.keys(): continue
        train_mask_tmp = cv2.cvtColor(test_mask[i], cv2.COLOR_BGR2GRAY)
        if np.sum(train_mask_tmp) == 0: continue
        # up
        if np.mean(train_mask_tmp[0]) < 255 and (np.sum(train_mask_tmp[0]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(y) - 1) in map_img[mapid][x].keys() and int(mapid) < 159:
                imgid = map_img[mapid][x][str(int(y) - 1)]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("down", train_mask_tmp[0].tolist())
        # down
        if np.mean(train_mask_tmp[100]) < 255 and (np.sum(train_mask_tmp[100]) > 255):
            # tmp rule
            if i in semi_vertical_mask.keys(): continue
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(y) + 1) in map_img[mapid][x].keys() and int(mapid) < 159:
                imgid = map_img[mapid][x][str(int(y) + 1)]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("up", train_mask_tmp[100].tolist())
        # left
        if np.mean(train_mask_tmp[:, 0]) < 255 and (np.sum(train_mask_tmp[:, 0]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(x) - 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) - 1)].keys() and int(mapid) < 159:
                imgid = map_img[mapid][str(int(x) - 1)][y]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("right", train_mask_tmp[:, 0].tolist())
        # right
        if np.mean(train_mask_tmp[:, 100]) < 255 and (np.sum(train_mask_tmp[:, 100]) > 255):
            mapid, x, y = img_map[i]['mapid'], img_map[i]['x'], img_map[i]['y']
            if str(int(x) + 1) in map_img[mapid].keys() and y in map_img[mapid][str(int(x) + 1)].keys() and int(mapid) < 159:
                imgid = map_img[mapid][str(int(x) + 1)][y]
                if imgid in test_set and np.sum(test_mask[imgid]) == 0: mask.setdefault(imgid, {}).setdefault("left", train_mask_tmp[:, 100].tolist())

    with open(r'./jigsaw/json_files/smooth_rule_on_test.json', 'w') as f:
        json.dump(mask, f)

    padding(r'./jigsaw/json_files/smooth_rule_on_test.json',
            r'./jigsaw/json_files/smooth_rule_on_test_test.json', img_map, map_img, test_mask)

def for_smooth_rule_on_test():
    test_mask_dict = {}
    with open(r'./jigsaw/json_files/smooth_rule_on_test_test.json') as f:
        rule6_dict = json.load(f)
    for k, v in rule6_dict.items():
        new_mask = np.zeros((101, 101))
        if len(v.keys()) == 1:
            continue
        if len(v.keys()) == 2:
            # up - down
            if 'up' in v.keys() and 'down' in v.keys():
                up_first, up_last = np.where(np.array(v['up'])==255)[0][0], \
                                    np.where(np.array(v['up'])==255)[0][-1]
                down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                        np.where(np.array(v['down']) == 255)[0][-1]
                k1, k2 = liner_line(up_first, 0, down_first, 100), liner_line(up_last, 0, down_last, 100)
                for i in range(new_mask.shape[0]):
                    new_mask[i, max(0, int(k1 * i + up_first)) : min(101, int(k2 * i + up_last))] = 255
            if 'left' in v.keys() and 'right' in v.keys():
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                k1 = liner_line(left_first, 0, right_first, 100)
                for i in range(new_mask.shape[1]):
                    new_mask[min(100, max(0, int(k1 * i + left_first))):, i] = 255

            if 'up' in v.keys() and 'left' in v.keys():
                continue
            if 'up' in v.keys() and 'right' in v.keys():
                continue
            if 'down' in v.keys() and 'left' in v.keys():
                if np.mean(v['down']) != 255:
                    down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                            np.where(np.array(v['down']) == 0)[0][0]
                else:
                    down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                            np.where(np.array(v['down']) == 255)[0][-1]
                left_first, left_last = np.where(np.array(v['left']) == 255)[0][0], \
                                        np.where(np.array(v['left']) == 255)[0][-1]
                k1 = liner_line(0, left_first, down_last, 100)
                for i in range(left_first, 101):
                    new_mask[i, 0 : min(101, int(k1 * (i - left_first)))] = 255
            if 'down' in v.keys() and 'right' in v.keys():
                if np.mean(v['down']) != 255:
                    down_first, down_last = np.where(np.array(v['down']) == 0)[0][-1], \
                                            np.where(np.array(v['down']) == 255)[0][-1]
                else:
                    down_first, down_last = np.where(np.array(v['down']) == 255)[0][0], \
                                            np.where(np.array(v['down']) == 255)[0][-1]
                right_first, right_last = np.where(np.array(v['right']) == 255)[0][0], \
                                          np.where(np.array(v['right']) == 255)[0][-1]
                k1 = liner_line(100, right_first, down_first, 100)
                for i in range(right_first, 101):
                    new_mask[i, max(0, int(k1 * (i - right_first) + 100)): 101] = 255
        if len(v.keys()) == 3:
            continue
        new_mask = new_mask[:, :, np.newaxis]
        if np.sum(new_mask) != 0:
            test_mask_dict[k] = np.reshape(new_mask, (101, 101))
    print("Smooth Rule1 for test detect : ", len(test_mask_dict.keys()))

    os.remove(r'./jigsaw/json_files/smooth_rule_on_test_test.json')
    return test_mask_dict

def make_sub_rule_smoth(test_mask_path):
    save_csv_path = test_mask_path.replace('.csv', '') + '-smooth.csv'

    test_mask_dict = load_test_mask(test_mask_path)
    test_mask_dict1 = for_smooth_rule()
    test_mask_dict2 = for_smooth_rule_on_test()
    test_mask_dict.update(test_mask_dict2)
    test_mask_dict.update(test_mask_dict1)
    out = []
    count = 0
    for id in test_mask_dict:
        if len(test_mask_dict[id].shape) == 3:
            sum_tmp = test_mask_dict[id][:, :, 0]
        else:
            sum_tmp = test_mask_dict[id]
        out.append([id, sum_tmp])
        count += 1
    submission = create_submission(out)
    submission.to_csv(save_csv_path, index=None)
    print(count)

def jigsaw_folder_dict(path):
    import json
    with open(path, 'r') as load_f:
        dict = json.load(load_f)

    print(len(dict))
    return dict

def make_vertical_rule_sub(jason_type_v1, jason_type_v2, train_mask_path, csv_path):
    save_csv_path = csv_path.replace('.csv', '') + '-vertical.csv'
    print(jason_type_v1)
    dict_jigsaw_vertical = jigsaw_folder_dict(jason_type_v1)
    print(jason_type_v2)
    dict_jigsaw_semi_vertical = jigsaw_folder_dict(jason_type_v2)

    dict = decode_csv(csv_name=csv_path)

    for id in dict_jigsaw_vertical:
        train_id = dict_jigsaw_vertical[id]
        train_mask_tmp = os.path.join(train_mask_path,train_id+'.png')
        train_mask = cv2.imread(train_mask_tmp,0) / 255
        dict[id] = train_mask

    for id in dict_jigsaw_semi_vertical:
        train_id = dict_jigsaw_semi_vertical[id]
        train_mask_tmp = os.path.join(train_mask_path,train_id+'.png')
        train_mask = cv2.imread(train_mask_tmp, 0)
        down_bound = train_mask[100,:]
        for i in range(101):
            train_mask[i,:] = down_bound
        train_mask = train_mask/ 255
        dict[id] = train_mask

    # print(count)
    out = []
    for id in dict:
        sum_tmp = dict[id]
        out.append([id, sum_tmp])
    submission = create_submission(out)
    submission.to_csv(save_csv_path, index=None)


def vertical_rule_sub(img_map, map_img, vertical_mask, semi_vertical_mask, submission_path):
    vertical_rule1(img_map, map_img, vertical_mask)
    vertical_rule2(vertical_mask, img_map, map_img)
    vertical_rule3(vertical_mask, submission_path, img_map, map_img)
    semi_vertical_rule(semi_vertical_mask, img_map, map_img)
    ### resolve rule conflict
    vertical_rules_combine(rule=['1', '2', '3_1', '3_2', '3_3', '3_4'])
    ### make submission on vertical rules
    make_vertical_rule_sub('./jigsaw/json_files/vertical.json',
                           './jigsaw/json_files/semi_vertical.json',
                           train_mask_path, submission_path)


def empty_rule_sub(img_map, map_img, img_map_set, submission_path):

    build_full_train_mask(train_mask_path, img_map, img_map_set)
    with open(r'./jigsaw/json_files/jigsaw_map_virtual.json') as f:
        img_map_virtual = json.load(f)
        map_img_virtual = {}
        for k, v in img_map_virtual.items():
            map_img_virtual.setdefault(v['mapid'], {}).setdefault(v['x'], {}).setdefault(v['y'], k)

    ### apply empty rules
    empty_rule1(train_mask_path, img_map, img_map_set, map_img)
    empty_rule2(train_mask_path, submission_path, img_map, img_map_set, map_img, map_img_virtual)
    ### make submission on empty rules
    make_sub_rule_empty(submission_path)

def smoth_rule_sub(train_mask_path,img_map, map_img, img_map_set, vertical_mask, semi_vertical_mask, submission_path):

    smooth_rule(train_mask_path, submission_path, vertical_mask, img_map, img_map_set, map_img)
    smooth_rule_on_test(submission_path, vertical_mask, semi_vertical_mask, img_map, img_map_set, map_img)
    ###make submission on smooth rules
    make_sub_rule_smoth(submission_path)


def submission_apply_jigsaw_postprocessing(submission_path):
    # step 1: find all vertical masks in train set and save to json
    find_all_vertical_mask()

    # step2: load all vertical masks in train set from json
    with open(r'./jigsaw/json_files/train_vertical_mask.json', 'r') as mf:
        vertical_mask = json.load(mf)
        print('num_full_vertical_masks: ', len(vertical_mask.keys()))

    with open(r'./jigsaw/json_files/train_semi_vertical_mask.json', 'r') as mf:
        semi_vertical_mask = json.load(mf)
        print('num_semi_vertical_masks: ', len(semi_vertical_mask.keys()))

    # step3: load jigsaw puzzles
    with open(r'./jigsaw/json_files/jigsaw_maps.json', 'r') as f:
        img_map = json.load(f)
        img_map_set = set(img_map.keys())
        map_img = {}
        for k, v in img_map.items():
            map_img.setdefault(v['mapid'], {}).setdefault(v['x'], {}).setdefault(v['y'], k)

    # apply vertical rules
    vertical_rule_sub(img_map, map_img, vertical_mask, semi_vertical_mask, submission_path)

    #apply empty rules
    submission_vertical_path = submission_path.replace('.csv', '') + '-vertical.csv'
    empty_rule_sub(img_map, map_img, img_map_set, submission_vertical_path)

    # apply smooth rules
    submission_empty_path = submission_vertical_path.replace('.csv', '') + '-empty.csv'
    smoth_rule_sub(train_mask_path, img_map, map_img, img_map_set, vertical_mask, semi_vertical_mask, submission_empty_path)

    os.remove(submission_vertical_path)
    os.remove(submission_empty_path)


#################### apply rules ####################
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='JIGSAW PUZZLE RULES')
    parser.add_argument('--submission_path', required=True, help='Path for predicted test masks')
    args = parser.parse_args()
    submission_path = args.submission_path
    submission_apply_jigsaw_postprocessing(submission_path)

