import logging
import boto3
from botocore.exceptions import ClientError
import os

from dotenv import load_dotenv
load_dotenv()

def download_file(file_name, bucketName='mlflow-bucket', object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """


    s3_resource = boto3.resource('s3', aws_access_key_id='minio-id', aws_secret_access_key='minio-key')
    bucket = s3_resource.Bucket(bucketName)
    # bucket.download_file(object_name, file_name)
    for bucket in s3_resource.buckets.all():
        print(bucket.name)

if __name__=="__main__":
    download_file(file_name='input_examples/asset_variety_new.json', object_name='1/8b1e6cd7ec2f4feaa1577f340c34f944/artifacts/model/input_example.json')