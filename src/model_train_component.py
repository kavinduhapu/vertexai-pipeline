from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Artifact

@dsl.component(base_image='python:3.11', packages_to_install=['pandas==2.2.2', 'scikit-learn==1.5.2'],
               target_image='gcr.io/my-project/my-component:v1')

def model_train_component(scaled_data: Input[Artifact]):
    import pickle
    
    from model_train.random_forest import train_rf_model
    
    with open (scaled_data.path, wb) as file:
        load_dict = pickle.load(file)
        
    X_train_scaled = load_dict['X_train_scaled']
    X_test_scaled = load_dict['X_test_scaled']
    y_train = load_dict['y_train']
    y_test = load_dict['y_test']
    
    train_rf_model(X_train_scaled, X_test_scaled, y_train, y_test)