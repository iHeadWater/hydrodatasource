"""Main module."""
import os.path

from minio import Minio

from hydroprivatedata import config


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
    bucket_names = [bucket.name for bucket in client.list_buckets()]
    if bucket_name not in bucket_names:
        client.make_bucket(bucket_name)
    # Upload an object
    client.fput_object(bucket_name, object_name, file_path)
    # List objects
    objects = client.list_objects(bucket_name, recursive=True)
    return [obj.object_name for obj in objects]


def minio_download_csv(client: Minio, bucket_name, object_name, file_path: str, version_id=None):
    try:
        response = client.get_object(bucket_name, object_name, version_id)
        res_csv: str = response.data.decode('utf8')
        with open(os.path.join(config.LOCAL_DATA_PATH, file_path+'.csv'), 'w+') as fp:
            fp.write(res_csv)
    finally:
        response.close()
        response.release_conn()


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
    bucket_names = [dic['Name'] for dic in client.list_buckets()['Buckets']]
    if bucket_name not in bucket_names:
        client.create_bucket(Bucket=bucket_name)
    # Upload an object
    client.upload_file(file_path, bucket_name, object_name)
    # List objects
    objects = [dic['Key'] for dic in client.list_objects(Bucket=bucket_name)['Contents']]
    return objects


def boto3_download_csv(client, bucket_name, object_name, file_path: str):
    client.download_file(bucket_name, object_name, file_path)
