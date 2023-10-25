"""Main module."""


def minio_upload_csv(client, bucket_name, object_name, file_path):
    """upload csv to minio

    Parameters
    ----------
    client : _type_
        the minio client
    bucket_name : _type_
        the bucket name
    object_name : _type_
        the object name
    file_path : _type_
        the local file path
    """
    # Make a bucket
    client.make_bucket(bucket_name)
    # Upload an object
    client.fput_object(bucket_name, object_name, file_path)
    # List objects
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        print(obj.object_name)


def boto3_upload_csv(client, bucket_name, object_name, file_path):
    """upload csv to minio

    Parameters
    ----------
    client : _type_
        the minio client
    bucket_name : _type_
        the bucket name
    object_name : _type_
        the object name
    file_path : _type_
        the local file path
    """
    # Make a bucket
    client.create_bucket(Bucket=bucket_name)
    # Upload an object
    client.upload_file(file_path, bucket_name, object_name)
    # List objects
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        print(obj.object_name)
