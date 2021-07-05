import numpy as np

def read_bci_data():
#     得到個別的資料
    S4b_train = np.load('S4b_train.npz') #  540筆資料，每筆共有750 time point 且有2個chanel (來自腦部中的兩個測試點) 
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')   #  540筆資料，每筆的label 是1 or 2 (左手 or 右手)
    X11b_test = np.load('X11b_test.npz')
#     把兩組大小540的資料合併成1080大小
    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0) 
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)
#    把label 換成0,1 
#    expand_dims擴展維度，下方用此方法把1080,750,2 在axis=1的地方讓它變成 1080,1,750,2
#    用transpose改變為度的順序 將 1080,1,750,2 透過0 1 3 2 順序轉換成 1080,1,2,750
    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
#    把資料中nan的位置找出來，並且放入nanmean處理 nanmean是跳過nan取mean
#     np.isnan()會檢查train_data每個位置是否為nan
#     再透過where把nan的位置找出來並存入mask,是一個有4個數組的tuple,每個數組的第1個數字組成的4維度位置就是第一個nan的地方
    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)
    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return train_data, train_label, test_data, test_label
