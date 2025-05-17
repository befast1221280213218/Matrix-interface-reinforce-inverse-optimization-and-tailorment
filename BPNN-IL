import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau 



epochs = 1 
epsilon = 1e-6  
random_states=j
loss_save_path = r'G:\Article_IIII_final\Code\Forwardprediction\Output\BPNN_epoch-loss{}.xlsx'.format(j)
metrics_save_path = r'G:\Article_IIII_final\Code\Forwardprediction\Output\BPNN_epoch-metrics{}.xlsx'.format(j)


file_paths = [
    r'data1.xlsx',
    r'data2.xlsx',
    r'data3.xlsx'
]


selected_columns = ['Fracture_strain','A','B','n','C',
                    'Volume_interface' ,'Interface_thickness', 'Al4C3','Al4Si3',
                    'Configurations', 'Diameter', 'Volume Fraction','Varience',
                    'UTS', 'Kt','E']
datasets = [pd.read_excel(file) for file in file_paths] 


normalization_factors = {col: max([df[col].max() for df in datasets]) for col in selected_columns}
for df in datasets:
    for col in selected_columns:
        df[col] /= normalization_factors[col]

print( normalization_factors["E"])

X = [df[selected_columns[:13]].values for df in datasets]
Y = [df[selected_columns[13:]].values for df in datasets]


X_all = np.vstack(X)
Y_all = np.vstack(Y)


num_samples = len(X_all)


train_size = int(num_samples * 0.5)
val_size = int(num_samples * 0.2)
test_size = num_samples - train_size - val_size  


X_train, X_temp, Y_train, Y_temp = train_test_split(X_all, Y_all, train_size=train_size, random_state=random_states)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size=val_size, random_state=random_states)



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(13,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3)  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    return model


model = build_model()
history = model.fit(X_train, Y_train, 
                    epochs=epochs, 
                    batch_size=32, 
                    validation_data=(X_val, Y_val), 
                    verbose=0)



def calculate_metrics(y_true, y_pred):
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100  
    }
    return metrics


Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)
Y_test_pred = model.predict(X_test)

print(Y_train_pred.shape)


metrics_results = {}

for dataset_name, Y_true, Y_pred in [("Train", Y_train, Y_train_pred),
                                     ("Validation", Y_val, Y_val_pred),
                                     ("Test", Y_test, Y_test_pred)]:
    dataset_metrics = {}
    for i, target_name in enumerate(['UTS', 'Kt','E']):
        dataset_metrics[target_name] = calculate_metrics(Y_true[:, i], Y_pred[:, i])
        print(f"{dataset_name} - {target_name}: {dataset_metrics[target_name]}")
    
    metrics_results[dataset_name] = dataset_metrics


metrics_dfs = {dataset: pd.DataFrame(metrics_results[dataset]).T for dataset in metrics_results}
final_metrics_df = pd.concat(metrics_dfs, axis=1)
final_metrics_df.columns = pd.MultiIndex.from_product([metrics_results.keys(), ["MSE", "RMSE", "R2", "MAPE"]])



##---------------------------------------------------------------Step2------------------------------------------------------#
print("\n--- IL ---")
results = [] 
loss_history = []  
r2_history = []  
epoch_loss_r2 = [] 

for i in range(len(datasets)):
    print(f" {i} ...")
    if i == 0:
        model = build_model()
    else:
        model.load_weights(f'incremental_model_{i-1}.weights.h5')
    
    
    X_train, X_temp, y_train, y_temp = train_test_split(X[i], Y[i], test_size=0.5, random_state=random_states)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.6, random_state=random_states)
    
   
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2000, min_delta=0.001, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10000, min_lr=1e-6)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping, reduce_lr])
    model.save_weights(f'incremental_model_{i}_{j}.weights.h5')
    
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)
    
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred, multioutput='raw_values')
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        return mse, rmse, mae, r2, mape
    
    train_metrics = evaluate(y_train, y_train_pred)
    test_metrics = evaluate(y_test, y_test_pred)
    val_metrics = evaluate(y_val, y_val_pred)
    
    results.append([ 
        ["Train", *train_metrics],
        ["Test", *test_metrics],
        ["Validation", *val_metrics]
    ])
    
    
    loss_history.append([history.history['loss'], history.history['val_loss']])
    
   
    r2_history.append([train_metrics[3].mean(), test_metrics[3].mean(), val_metrics[3].mean()])
    
   
    for epoch in range(len(history.history['loss'])):
        epoch_loss_r2.append([i, epoch, history.history['loss'][epoch], history.history['val_loss'][epoch], 
                             train_metrics[3].mean(), test_metrics[3].mean(), val_metrics[3].mean()])


results_df = pd.DataFrame(
    sum(results, []),
    columns=["Dataset", "MSE", "RMSE", "MAE", "R2", "MAPE"]
)
results_df.to_excel(r'G:\Article_IIII_final\Code\Forwardprediction\Output\Incremental_Learning_Results_{}.xlsx'.format(j), index=False)
