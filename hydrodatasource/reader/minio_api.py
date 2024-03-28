"""Main module."""
import asyncio
import os
import pathlib

from minio import Minio
import chardet


async def minio_upload_csv(client, bucket_name, object_name, file_path):
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


async def minio_download_csv(client: Minio, bucket_name, object_name: str, file_path: str, version_id=None):
    try:
        response = client.get_object(bucket_name, object_name, version_id)
        # 图片不能直接解码, 但可以直接对文件写入response.data
        encoding = chardet.detect(response.data)['encoding']
        object_path = os.path.join(file_path, object_name)
        object_parent = pathlib.Path(object_path).parent
        if not object_parent.exists():
            object_parent.mkdir(parents=True)
        if encoding is not None:
            res_csv: str = response.data.decode(encoding)
            with open(object_path, 'w+', encoding=encoding) as fp:
                fp.write(res_csv)
        else:
            with open(object_path, 'wb') as fp:
                fp.write(response.data)
    finally:
        response.close()
        response.release_conn()


async def boto3_upload_csv(client, bucket_name, object_name, file_path):
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


async def boto3_download_csv(client, bucket_name, object_name, file_path: str):
    client.download_file(bucket_name, object_name, file_path)


async def boto3_sync_files(client, bucket_name, local_path, bucket_path=None):
    """
    :param client: the boto3 client
    :param bucket_name: the bucket name which you want to sync your data
    :param local_path: the path on your local machine
    :param bucket_path: the path under your bucket which you want to sync
    :return:
    """
    remote_objects = [dic['Key'] for dic in client.list_objects(Bucket=bucket_name)['Contents']]
    local_objects = [str(path.relative_to(local_path)).replace('\\', '/')
                     for path in pathlib.Path(local_path).rglob(pattern='*') if path.is_file()]
    objects_in_remote = [obj for obj in remote_objects if obj not in local_objects]
    objects_in_local = [obj for obj in local_objects if obj not in remote_objects]
    task_batch_dload = asyncio.create_task(boto3_batch_download(objects_in_remote, client, bucket_name, local_path))
    task_batch_upload = asyncio.create_task(boto3_batch_upload(objects_in_local, client, bucket_name, local_path))
    await asyncio.gather(task_batch_upload, task_batch_dload)


async def minio_batch_download(objects_in_remote, client: Minio, bucket_name, local_path):
    # 将下载任务打包成大量task容易造成许多上下文切换进而挤兑，目前看来不如顺序下载
    for obj in objects_in_remote:
        await minio_download_csv(client, bucket_name, obj, local_path)


async def minio_batch_upload(objects_in_local, client: Minio, bucket_name, local_path):
    for obj in objects_in_local:
        await minio_upload_csv(client, bucket_name, obj, local_path)


async def boto3_batch_download(objects_in_remote, client, bucket_name, local_path):
    # 将下载任务打包成大量task容易造成许多上下文切换进而挤兑，目前看来不如顺序下载
    for obj in objects_in_remote:
        await boto3_download_csv(client, bucket_name, obj, local_path)


async def boto3_batch_upload(objects_in_remote, client, bucket_name, local_path):
    for obj in objects_in_remote:
        await boto3_upload_csv(client, bucket_name, obj, local_path)


async def minio_sync_files(client: Minio, bucket_name, local_path, bucket_path=None):
    """
    :param client: the minio client
    :param bucket_name: the bucket name which you want to sync your data
    :param local_path: the path on your local machine, contents of bucket_name will be saved to the directory directly
    :param bucket_path: the path under your bucket which you want to sync
    :return:
    """
    remote_objects = [obj.object_name for obj in client.list_objects(bucket_name, prefix=bucket_path, recursive=True)]
    local_objects = [str(path.relative_to(local_path)).replace('\\', '/')
                     for path in pathlib.Path(local_path).rglob(pattern='*') if path.is_file()]
    objects_in_remote = [obj for obj in remote_objects if obj not in local_objects]
    objects_in_local = [obj for obj in local_objects if obj not in remote_objects]
    task_batch_dload = asyncio.create_task(minio_batch_download(objects_in_remote, client, bucket_name, local_path))
    task_batch_upload = asyncio.create_task(minio_batch_upload(objects_in_local, client, bucket_name, local_path))
    await asyncio.gather(task_batch_upload, task_batch_dload)
