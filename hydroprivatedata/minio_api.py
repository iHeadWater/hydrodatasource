"""Main module."""
import os

from minio import Minio


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
        with open(file_path, 'w+') as fp:
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


def boto3_sync_files(client, bucket_name, local_path, bucket_path=None):
    """
    :param client: the boto3 client
    :param bucket_name: the bucket name which you want to sync your data
    :param local_path: the path on your local machine
    :param bucket_path: the path under your bucket which you want to sync
    :return:
    """
    remote_objects = [dic['Key'] for dic in client.list_objects(Bucket=bucket_name)['Contents']]
    local_objects = os.scandir(local_path)
    objects_in_remote = [obj for obj in remote_objects if obj not in local_objects]
    objects_in_local = [obj for obj in local_objects if obj not in remote_objects]
    for obj in objects_in_remote:
        local_obj_path = os.path.join(local_path, obj)
        client.download_file(bucket_name, obj, local_obj_path)
    for obj in objects_in_local:
        remote_obj = obj
        client.upload_file(obj, bucket_name, remote_obj)


def minio_sync_files(client: Minio, bucket_name, local_path, bucket_path=None):
    """
    :param client: the minio client
    :param bucket_name: the bucket name which you want to sync your data
    :param local_path: the path on your local machine
    :param bucket_path: the path under your bucket which you want to sync
    :return:
    """
    # 考虑根据不同情况设置recursive=True
    remote_objects = client.list_objects(bucket_name)
    local_objects = os.scandir(local_path)
    objects_in_remote = [obj for obj in remote_objects if obj not in local_objects]
    objects_in_local = [obj for obj in local_objects if obj not in remote_objects]
    for obj in objects_in_remote:
        local_obj_path = os.path.join(local_path, obj)
        minio_download_csv(client, bucket_name, obj, local_obj_path)
    for obj in objects_in_local:
        remote_obj = obj
        minio_upload_csv(client, bucket_name, remote_obj, local_path)
