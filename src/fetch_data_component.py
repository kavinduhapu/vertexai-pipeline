from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model

@dsl.component(base_image='python:3.11', packages_to_install=['pandas==2.2.2'],
               target_image='gcr.io/my-project/my-component:v1')

def fetch_data_component(iris_data: Output[Dataset]):
    from fetch_data.fetch_data import load_iris
    iris_data_df = load_iris()
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # Add target column
    df['y'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    print(df.head())
    df.to_csv(iris_data.path)