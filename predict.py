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
    model = load_model('/model_dense_slicing.keras')

    with open(f'sclrX', 'rb') as f:
        sclrX = pickle.load(f)
        f.close

    x_test = sclrX.transform(x_test)
    return model.predict(x_test).squeeze()
