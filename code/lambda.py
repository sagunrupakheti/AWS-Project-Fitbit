import time
import boto3
import pandas as pd
from io import StringIO
from time import gmtime, strftime

# Initialize the boto3 client for SageMaker and S3
s3 = boto3.client('s3')
sagemaker = boto3.client('sagemaker') 

runtime = boto3.client('sagemaker-runtime')

# Specify your details
bucket_name = 'sagunprojectbucket'
file_key = 'sagemaker/calorie_prediction/predictions/full_dataset.csv'
model_name = 'Custom-sklearn-model-2024-04-18-02-14-56'
endpoint_config_name = 'my-endpoint-sagun'
endpoint_name = 'sagun-endpoint'

def get_csv_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

def create_endpoint(model_name, endpoint_config_name, endpoint_name):
    try:
        # Create endpoint configuration
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.t2.medium',
                'InitialVariantWeight': 1
            }]
        )
        # Create endpoint
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        print("Endpoint is being created...")
        sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
        print("Endpoint is in service.")
    except Exception as e:
        print(e)

def wait_for_endpoint_to_be_ready(endpoint_name):
    """Poll endpoint status until it is in service"""
    status = check_endpoint_status(endpoint_name)
    while status not in ['InService', 'Failed']:
        print(f"Waiting for endpoint to be ready... Current status: {status}")
        time.sleep(60)  # wait for 60 seconds before checking again
        print('Checking')
        status = check_endpoint_status(endpoint_name)
    return status
    
def check_endpoint_status(endpoint_name):
    response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    return response['EndpointStatus']

def predict(endpoint_name, data):
    payload = data.to_csv(header=False, index=False).encode('utf-8')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    return pd.Series(response['Body'].read().decode('utf-8').splitlines())

def delete_endpoint(endpoint_name, endpoint_config_name):
    sagemaker.delete_endpoint(EndpointName=endpoint_name)
    sagemaker.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print("Endpoint and its configuration have been deleted.")

def save_predictions_to_s3(bucket, key, data_frame):
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"Modified CSV has been saved to {bucket}/{key}")

def save_csv_to_s3(bucket, key, data_frame):
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"Modified CSV has been saved to {bucket}/{key}.")

# Workflow
data = get_csv_from_s3(bucket_name, file_key)
print(data)
data.drop(columns=['Id', 'ActivityDate','Calculated Calories'],inplace=True)
print('dropped')
# delete_endpoint(endpoint_name, endpoint_config_name)
create_endpoint(model_name, endpoint_config_name, endpoint_name)
print('endpoint created')
status = wait_for_endpoint_to_be_ready(endpoint_name)
print('waiting')
if status == 'InService':
    print('prediction started')
    data['Prediction_calories'] = predict(endpoint_name, data)
    print('tocsv')
    save_csv_to_s3(bucket_name, 'calorie-predictions' , data)
    delete_endpoint(endpoint_name, endpoint_config_name)

