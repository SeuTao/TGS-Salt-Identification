from include import *

Disimi_Threshold = 10
Compa_Threshold = 0.2

train_img_path = os.path.join(root_dir,'train/images/')
train_mask_path = os.path.join(root_dir,'train/masks/')
test_path = os.path.join(root_dir,'test/images/')
train_file_path = os.path.join(root_dir,'train.csv')
test_file_path = os.path.join(root_dir,'test.csv')

def make_test_list():
    outfile = open(test_file_path, 'w')
    print(test_path)
    for subdir in os.listdir(test_path):
        outfile.write(subdir + '\n')
    outfile.close()

def make_train_files(train_file_path):
    train_files = []
    infile = open(train_file_path, 'r')
    for line in infile:
        line = line.strip()
        row = line.split(',')
        if row[0] == 'id':
            continue
        train_files.append(row[0] + '.png')
    return np.asarray(train_files)

def make_test_files(test_file_path):
    test_files = []
    infile = open(test_file_path, 'r')
    for line in infile:
        line = line.strip()
        row = line.split(',')
        if row[0] == 'id':
            continue
        test_files.append(row[0])
    return np.asarray(test_files)

make_test_list()

train_files = make_train_files(train_file_path)
test_files = make_test_files(test_file_path)
all_files = np.append(train_files,test_files, axis=0)

def load_img(img_path):
    img = cv2.imread(img_path)
    return img

img_train = np.zeros((4000, 101, 101))
mask_train = np.zeros((4000, 101, 101))
for i, file in enumerate(train_files):
    img_train[i] = np.expand_dims(np.array(load_img(train_img_path + file))[:, :, 0], axis=0)
    mask_train[i] = np.expand_dims(np.array(load_img(train_mask_path + file))[:, :, 0], axis=0)

img_test = np.zeros((18000, 101, 101))
for i, file in enumerate(test_files):
    img_test[i] = np.expand_dims(np.array(load_img(test_path + file))[:, :, 0], axis=0)

all_arr = np.append(img_train, img_test, axis=0)
all_arr = all_arr / 255.0
del img_train, img_test
gc.collect()

all_u_ex = 2*all_arr[:, :, 0] - all_arr[:, :, 1]
all_u_ex = (all_u_ex - np.mean(all_u_ex, axis=0)) / np.std(all_u_ex, axis=0, ddof=1)

all_d_ex = 2*all_arr[:, :, 100]-all_arr[:, :, 99]
all_d_ex = (all_d_ex - np.mean(all_d_ex, axis=0)) / np.std(all_d_ex, axis=0, ddof=1)

all_l_ex = 2*all_arr[:, 0, :]-all_arr[:, 1, :]
all_l_ex = (all_l_ex - np.mean(all_l_ex, axis=0)) / np.std(all_l_ex, axis=0, ddof=1)

all_r_ex = 2*all_arr[:, 100, :]-all_arr[:, 99, :]
all_r_ex = (all_r_ex - np.mean(all_r_ex, axis=0)) / np.std(all_r_ex, axis=0, ddof=1)

all_u_ex = np.transpose((np.transpose(all_u_ex) - np.mean(all_u_ex, axis=1)) / np.std(all_u_ex, axis=1, ddof=1))
all_d_ex = np.transpose((np.transpose(all_d_ex) - np.mean(all_d_ex, axis=1)) / np.std(all_d_ex, axis=1, ddof=1))
all_l_ex = np.transpose((np.transpose(all_l_ex) - np.mean(all_l_ex, axis=1)) / np.std(all_l_ex, axis=1, ddof=1))
all_r_ex = np.transpose((np.transpose(all_r_ex) - np.mean(all_r_ex, axis=1)) / np.std(all_r_ex, axis=1, ddof=1))

tree = KDTree(all_u_ex)
dist_ud, ind_ud = tree.query(all_d_ex, k=2)

tree = KDTree(all_d_ex)
dist_du, ind_du = tree.query(all_u_ex, k=2)

tree = KDTree(all_l_ex)
dist_lr, ind_lr = tree.query(all_r_ex, k=2)

tree = KDTree(all_r_ex)
dist_rl, ind_rl = tree.query(all_l_ex, k=2)

from pandas import Series,DataFrame

def gen_mosaic(ad, ac):
    # generate candidates left-right
    length = dist_lr.shape[0]
    dlr_dict = {'i1': ind_lr[:, 0],
                'i2': ind_lr[:, 1],
                'd1': dist_lr[:, 0],
                'd2': dist_lr[:, 1],
                'i0': np.asarray([i for i in range(length)]),
                'c': 1 - dist_lr[:, 0] / dist_lr[:, 1]}
    dlr = DataFrame(dlr_dict)

    drl_dict = {'i1': ind_rl[:, 0],
                'i2': ind_rl[:, 1],
                'd1': dist_rl[:, 0],
                'd2': dist_rl[:, 1],
                'i0': np.asarray([i for i in range(length)]),
                'c': 1 - dist_rl[:, 0] / dist_rl[:, 1]}
    drl = DataFrame(drl_dict)

    bb = pd.merge(dlr, drl, how='inner', left_on='i0', right_on='i1')
    # filter by disimilarity and compatibility
    bb2 = bb.loc[(bb.i0_x != bb.i1_x) & (bb.d1_x < ad) & (bb.c_x > ac) & (bb.c_y > ac)]
    # find left-right strips
    nt = length
    lcols = []
    eval = np.zeros(nt, dtype=int) > 0
    for i in range(length):
        if not eval[i]:
            lt = np.array([i])
            eval[i] = False
            cond = True
            i0 = i
            while (cond):
                i1 = np.array(bb2[bb.i0_x == i0]['i1_x'])
                if len(i1) == 1 and np.where(lt == i1[0])[0].shape[0] == 0:
                    lt = np.concatenate((i1, lt))
                    i0 = i1[0]
                    eval[i1[0]] = True
                else:
                    cond = False
            cond = True
            i0 = i
            while (cond):
                i1 = np.array(bb2[bb.i1_x == i0]['i0_x'])
                if len(i1) == 1 and np.where(lt == i1[0])[0].shape[0] == 0:
                    lt = np.concatenate((lt, i1))
                    i0 = i1[0]
                    eval[i1[0]] = True
                else:
                    cond = False
            if lt.shape[0] > 1:
                lcols.append(lt.tolist())

    # same for up-down
    length = dist_du.shape[0]
    ddu_dict = {'i1': ind_du[:, 0],
                'i2': ind_du[:, 1],
                'd1': dist_du[:, 0],
                'd2': dist_du[:, 1],
                'i0': np.asarray([i for i in range(length)]),
                'c': 1 - dist_du[:, 0] / dist_du[:, 1]}
    ddu = DataFrame(ddu_dict)

    dud_dict = {'i1': ind_ud[:, 0],
                'i2': ind_ud[:, 1],
                'd1': dist_ud[:, 0],
                'd2': dist_ud[:, 1],
                'i0': np.asarray([i for i in range(length)]),
                'c': 1 - dist_ud[:, 0] / dist_ud[:, 1]}
    dud = DataFrame(dud_dict)

    bb = pd.merge(ddu, dud, how='inner', left_on='i0', right_on='i1')
    # filter by disimilarity and compatibility
    bb = bb.loc[(bb.i0_x != bb.i1_x) & (bb.d1_x < ad) & (bb.c_x > ac) & (bb.c_y > ac)]

    # Generate up-down strips
    nt = length
    lrows = []
    eval = np.zeros(nt, dtype=int) > 0
    for i in range(length):
        if not eval[i]:
            lt = np.array([i])
            eval[i] = False
            cond = True
            i0 = i
            while (cond):
                i1 = np.array(bb[bb.i0_x == i0]['i1_x'])
                if len(i1) == 1 and np.where(lt == i1[0])[0].shape[0] == 0:
                    lt = np.concatenate((i1, lt))
                    i0 = i1[0]
                    eval[i1[0]] = True
                else:
                    cond = False
            cond = True
            i0 = i
            while (cond):
                i1 = np.array(bb[bb.i1_x == i0]['i0_x'])
                if len(i1) == 1 and np.where(lt == i1[0])[0].shape[0] == 0:
                    lt = np.concatenate((lt, i1))
                    i0 = i1[0]
                    eval[i1[0]] = True
                else:
                    cond = False
            if lt.shape[0] > 1:
                lrows.append(lt.tolist())
    # Finally combine rows and colums
    rc = - np.ones(shape=(nt, 2), dtype=int)
    for i in range(len(lrows)):
        for j in lrows[i]:
            rc[j, 0] = i
    for i in range(len(lcols)):
        for j in lcols[i]:
            rc[j, 1] = i

    bt = pd.concat([bb, bb2], ignore_index=True)

    nodes_set = set()
    nodes_list = []
    edge_set = set()
    edge_list = []
    for i in range(bt.shape[0]):
        nodes_set.add(bt['i0_x'][i])
        nodes_set.add(bt['i1_x'][i])
        edge_list.append((bt['i0_x'][i], bt['i1_x'][i]))
    #     edge_set.add((bt['i0_x'][i], bt['i1_x'][i]))
    for i in nodes_set:
        nodes_list.append(i)

    G = nx.Graph(edge_list)
    clu = nx.connected_components(G)

    ls_tmp = {}
    count = 0
    for i in clu:
        count += 1
        tmp = sorted(i)
        key = tmp[0]
        value = tmp
        ls_tmp.setdefault(key, value)
    print(count)

    ls = []
    for i in range(length):
        if i in ls_tmp.keys():
            ls.append(ls_tmp[i])
        else:
            ls.append([i])

    lls = []
    for i in ls:
        lls.append(len(i))

    dd = {}
    for i in range(len(lls)):
        dd.setdefault(i, lls[i])

    dd = sorted(dd.items(), key=lambda item: item[1], reverse=True)
    return dd, ls, rc, lrows, lcols

dd, ls, rc, lrows, lcols = gen_mosaic(Disimi_Threshold, Compa_Threshold)
# Complete iteratively a mosaic from a seed image
def complete(se, mat):
    ir0 = rc[se, 0]
    ic0 = rc[se, 1]

    if len(np.where(mat[:, 0] == se)[0]) > 0:
        x0 = mat[np.where(mat[:, 0] == se)[0], 1]
        y0 = mat[np.where(mat[:, 0] == se)[0], 2]
    else:
        x0 = mat[0, 1]
        y0 = mat[0, 2]

    if ir0 > -1:
        r0 = lrows[ir0]
        # r0 = r0[::-1]
        for i in range(len(r0)):
            if len(np.where(mat[:, 0] == r0[i])[0]) == 0:
                mat_tmp = np.array([[r0[i], x0 - np.where(r0 == se)[0] + i, y0, -1]])
                mat = np.concatenate((mat, mat_tmp), axis=0)

    if ic0 > -1:
        c0 = lcols[ic0]
        c0 = c0[::-1]
        for i in range(len(c0)):
            if len(np.where(mat[:, 0] == c0[i])[0]) == 0:
                mat_tmp = np.array([[c0[i], x0, y0 - np.where(c0 == se)[0] + i, -1]])
                mat = np.concatenate((mat, mat_tmp), axis=0)
    mat[np.where(mat[:, 0] == se)[0][0], 3] = 1
    # mat = np.unique(mat, axis=0)
    return mat

def gen_mos(se):
    mat = -np.ones(shape=(1, 4), dtype=int)
    mat[0, 0] = se
    while np.sum(mat[:, 3]) < mat.shape[0]:
        for i in range(mat.shape[0]):
            if mat[i, 3] == -1:
                mat = complete(mat[i, 0], mat)
    mat[:, 1] = mat[:, 1] - np.min(mat[:, 1])
    mat[:, 2] = mat[:, 2] - np.min(mat[:, 2])
    return mat

####### generate jigsaw puzzles and save to files #######
def solve_jigsaw_puzzles(dir):
    jigsaw_puzzles = {}
    if not os.path.exists(os.path.join(dir,'jigsaw_file')):
        os.mkdir(os.path.join(dir,'jigsaw_file'))
    if not os.path.exists(os.path.join(dir,'json_files')):
        os.mkdir(os.path.join(dir,'json_files'))
    for ii in range(0, 514):
        mat = gen_mos(ls[dd[ii][0]][0])
        var1 = mat[:, 0]
        var1_name = []
        for i in var1:
            var1_name.append(all_files[i].split('.')[0])
        var1_name = np.asarray(var1_name)
        var2 = mat[:, 1]
        var3 = mat[:, 2]
        df1 = DataFrame({'id': var1_name, 'x': var2, 'y': var3})
        df1.to_csv(os.path.join(dir, r'jigsaw_file/jigsaw-' + str(ii) + '.csv'), index=False, index_label=False, header=False)
        for i in range(len(var1_name)):
            jigsaw_puzzles.setdefault(var1_name[i], {'x': str(mat[i, 1]), 'y': str(mat[i, 2]), 'mapid': str(ii)})
        with open(os.path.join(dir,r'json_files/jigsaw_maps.json'), 'w') as f:
            json.dump(jigsaw_puzzles, f)


