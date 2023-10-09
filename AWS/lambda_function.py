#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""This script contains the code for the Lambda Funcion"""


# **It is important to set the endpoint as an environment variables and give the lambda role permission to invoke endpoint in Sagemaker**
# 
# 
# ![data](images/lambda1.png)

# In[ ]:


import json
import boto3
import base64
import os

# Environment variables
endpoint_name = os.environ["ENDPOINT_NAME"] 
runtime = boto3.client("runtime.sagemaker")

def lambda_handler(event, context):
    
    # Get the encoded image
    encoded_image_data = event["image"]
    
    decoded_image = base64.b64decode(encoded_image_data)
    
    image_byte_array = bytearray(decoded_image)

    # Get model detections passing the endode image
    detections = get_detections(image_byte_array)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'detections': detections
        })
    }

def get_detections(image):
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType="image/jpeg", 
        Body=image
    )
    
    results = json.loads(response["Body"].read())
    detections = results["prediction"]

