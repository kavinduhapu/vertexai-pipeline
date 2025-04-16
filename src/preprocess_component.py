from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

@dsl.component(base_image='python:3.11', packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.2'],
               target_image='gcr.io/my-project/my-component:v1')

def preprocess_component(iris_data: Input[Dataset], scaled_data: Output[Artifact]):
    import pickle
    
    import pandas as pd
    
    from preprocess.split_data import train_test_split_data
    from preprocess.scaling import standard_scaling
   
    df = pd.read_csv(iris_data.path)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    
    X_train_scaled, X_test_scaled = standard_scaling(X_train, X_test)
    
    result_dict = {'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled, 'y_train': y_train, 'y_test':y_test}
    
    save_apth = scaled_data.path + "/scaled_data.pkl"
    with open(scaled_data.path, 'wb') as file:
        pickle.save(result_dict, file)
    