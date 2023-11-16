from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/syedvaseembasha/Customer_Mlops/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:/home/syedvaseembasha/.config/zenml/local_stores/dce8850f-c82f-4089-a469-2c918a7a56ce/mlruns" 