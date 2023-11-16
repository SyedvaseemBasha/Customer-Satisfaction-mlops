from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/home/syedvaseembasha/Customer_Mlops/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:/home/syedvaseembasha/.config/zenml/local_stores/dce8850f-c82f-4089-a469-2c918a7a56ce/mlruns"

# python3 -m venv venv
# source newmlops/bin/activate
#  python3 run_pipeline.py


# zenml integration install mlflow -y
# zenml experiment-tracker register mlflow_tracker_Customer_Mlops --flavor=mlflow
# zenml model-deployer register mlflow_Customer_Mlops --flavor=mlflow
# zenml stack register mlflow_stack_Customer_Mlops -a default -o default -d mlflow -e mlflow_tracker_Customer_Mlops --set

#  python3 run_deployment.py --config deploy
# python3 run_deployment.py --config predict


# zenml experiment-tracker register mlflow_the_newone --flavor=mlflow
#  zenml model-deployer register mlflow3 --flavor=mlflow

# zenml stack register mlflow_the_newone -a default -o default -d mlflow3 -e mlflow_tracker --set

# streamlit run streamlit_app.py