
def predict(x):
    '''
    x - файл .csv
    Нулевая колонка не используется
    Далее 8 параметров (начиная с индекса 1) в порядке: GGKP, GK, PE, DS, DTR, Wi, BK, BMK
    '''
    import pandas as pd
    import numpy as np
    import pickle
    from tensorflow.keras.models import load_model
    
    df_test = pd.read_csv(x, decimal=',' )
    x_test = df_test.iloc[:,[1,2,3,4,5,6,7,8]].values.astype(np.float32)
    kpef_test = df_test['KPEF'].values.reshape(-1,1)
    model = load_model('/content/model_K.keras')

    with open(f'/content/sclrX', 'rb') as f:
        sclrX = pickle.load(f)
        f.close
        
    with open(f'/content/sclrX_clip_1500_800', 'rb') as f:
        sclrK = pickle.load(f)
        f.close

    x_test = sclrX.transform(x_test)
    kpef_test = sclrK.transform(kpef_test)
    return model.predict([x_test, kpef_test]).squeeze()
