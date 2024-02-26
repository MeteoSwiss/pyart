import os

import boto3

BUCKET='pyart'
ENDPOINT_URL='https://eu-central-1.linodeobjects.com'
def download_metranet_lib():
    linode_obj_config = {
        "aws_access_key_id": os.environ['AWS_ACCESS_KEY_ID'],
        "endpoint_url": ENDPOINT_URL,
        'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY']}

    client = boto3.client("s3", **linode_obj_config)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(current_directory, 'lib')
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)

    # List objects in the bucket
    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=BUCKET):
        if 'Contents' in result:
            for obj in result['Contents']:
                # Get the object key
                object_key = obj['Key']
                # Extract file name from object key
                file_name = os.path.basename(object_key)
                # Construct local file path
                local_file_path = os.path.join(lib_dir, file_name)
                # Download the object
                client.download_file(BUCKET, object_key, local_file_path)
                print(f"Downloaded {object_key} to {local_file_path}")

if __name__ == '__main__':
    download_metranet_lib()
