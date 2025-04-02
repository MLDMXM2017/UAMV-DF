import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

FLD_idx_file='../data/image_name_with_FLD_label.csv'
n_fold=50
dst_path='../50_folds/'

if __name__==__name__:
    img_df = pd.read_csv(FLD_idx_file, encoding='utf_8')
    img_x = np.array(img_df['img_names'])
    img_y = np.array(img_df['FLD'])
    int_y=[int(y) for y in img_y]
    img_y = np.array(int_y)

    pos_y=[y for y in img_y if y==1]
    neg_y=[y for y in img_y if y==0]
    print("all samples num",len(img_y))
    print("pos samples num",len(pos_y))
    print("neg samples num",len(neg_y))

    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    cv = [(t, v) for (t, v) in skf.split(img_x, img_y)]
    for k in range(n_fold):
        train_id, test_id = cv[k]

        x_train, y_train = img_x[train_id], img_y[train_id]
        x_test, y_test = img_x[test_id], img_y[test_id]

        np.save(dst_path + f"train_fold{k}_x.npy", x_train)
        np.save(dst_path + f"train_fold{k}_y.npy", y_train)
        np.save(dst_path + f"test_fold{k}_x.npy", x_test)
        np.save(dst_path + f"test_fold{k}_y.npy", y_test)

        np.save(dst_path + f"train_fold{k}_id.npy", train_id)
        np.save(dst_path + f"test_fold{k}_id.npy", test_id)