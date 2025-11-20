"""
Author: Jianfeng Zhu
Date: 2023-10-25 18:49:02
LastEditTime: 2025-10-31 11:28:17
LastEditors: Wenyu Ouyang
Description: Some configs for minio server
FilePath: \hydrodatasource\hydrodatasource\configs\config.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""

import os
from pathlib import Path
import boto3
import s3fs
import yaml
from minio import Minio
import psycopg2


def read_setting(setting_path):
    if not os.path.exists(setting_path):
        raise FileNotFoundError(f"Configuration file not found: {setting_path}")

    with open(setting_path, "r", encoding="utf-8") as file:  # 指定编码为 UTF-8
        setting = yaml.safe_load(file)

    example_setting = (
        "minio:\n"
        "  server_url: 'http://minio.waterism.com:9090' # Update with your URL\n"
        "  client_endpoint: 'http://minio.waterism.com:9000' # Update with your URL\n"
        "  access_key: 'your minio access key'\n"
        "  secret: 'your minio secret'\n\n"
        "local_data_path:\n"
        "  root: 'D:\\data\\waterism' # Update with your root data directory\n"
        "  datasets-origin: 'D:\\data\\waterism\\datasets-origin'\n"
        "  datasets-interim: 'D:\\data\\waterism\\datasets-interim'\n"
        "  cache: 'D:\\data\\waterism\\.cache'\n"
        "postgres:\n"
        "  server_url: your_postgres_server_url\n"
        "  port: 5432\n"
        "  username: your_postgres_username\n"
        "  password: your_postgres_secret_code\n"
        "  database: your_postgres_database\n"
    )

    if setting is None:
        raise ValueError(
            f"Configuration file is empty or has invalid format.\n\nExample configuration:\n{example_setting}"
        )

    # Define the expected structure
    expected_structure = {
        "minio": ["server_url", "client_endpoint", "access_key", "secret"],
        "local_data_path": ["root", "datasets-origin", "datasets-interim", "cache"],
        "postgres": ["server_url", "port", "username", "password", "database"],
    }

    # Validate the structure
    try:
        for key, subkeys in expected_structure.items():
            if key not in setting:
                raise KeyError(f"Missing required key in config: {key}")

            if isinstance(subkeys, list):
                for subkey in subkeys:
                    if subkey not in setting[key]:
                        raise KeyError(f"Missing required subkey '{subkey}' in '{key}'")
    except KeyError as e:
        raise ValueError(
            f"Incorrect configuration format: {e}\n\nExample configuration:\n{example_setting}"
        ) from e

    return setting


SETTING_FILE = os.path.join(Path.home(), "hydro_setting.yml")
try:
    SETTING = read_setting(SETTING_FILE)
    LOCAL_DATA_PATH = SETTING["local_data_path"]["root"]
    CACHE_DIR = SETTING["local_data_path"]["cache"]
except (ValueError, FileNotFoundError) as e:
    LOCAL_DATA_PATH = Path.home().joinpath("hydrodatasource_data")
    CACHE_DIR = Path.home().joinpath("hydrodatasource_data", ".cache")
    SETTING = {
        "minio": {
            "server_url": "",
            "client_endpoint": "",
            "access_key": "",
            "secret": "",
        },
        "postgres": {
            "server_url": "",
            "port": 5432,
            "username": "",
            "password": "",
            "database": "",
        },
        "local_data_path": {
            "root": LOCAL_DATA_PATH,
            "cache": CACHE_DIR,
        }
    }
    print(e)
except Exception as e:
    LOCAL_DATA_PATH = Path.home().joinpath("hydrodatasource_data")
    CACHE_DIR = Path.home().joinpath("hydrodatasource_data", ".cache")
    SETTING = {
        "minio": {
            "server_url": "",
            "client_endpoint": "",
            "access_key": "",
            "secret": "",
        },
        "postgres": {
            "server_url": "",
            "port": 5432,
            "username": "",
            "password": "",
            "database": "",
        },
        "local_data_path": {
            "root": LOCAL_DATA_PATH,
            "cache": CACHE_DIR,
        }
    }
    print(f"Unexpected error: {e}")

# Initialize remote service settings
MINIO_PARAM = {}
RO = {}
S3 = None
MC = None
PS = None
FS = None

STATION_BUCKET = "stations"
STATION_OBJECT = "sites.csv"
GRID_INTERIM_BUCKET = "grids-interim"

# Handle MinIO connection
try:
    # MinIO parameters
    MINIO_PARAM = {
        "endpoint_url": SETTING["minio"]["client_endpoint"],
        "key": SETTING["minio"]["access_key"],
        "secret": SETTING["minio"]["secret"],
    }

    # Initialize S3 FileSystem
    FS = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": MINIO_PARAM["endpoint_url"]},
        key=MINIO_PARAM["key"],
        secret=MINIO_PARAM["secret"],
        use_ssl=False,
    )

    # remote_options parameters for xr open_dataset from minio
    RO = {
        "client_kwargs": {"endpoint_url": MINIO_PARAM["endpoint_url"]},
        "key": MINIO_PARAM["key"],
        "secret": MINIO_PARAM["secret"],
        "use_ssl": False,
    }

    # Set up MinIO client
    S3 = boto3.client(
        "s3",
        endpoint_url=SETTING["minio"]["client_endpoint"],
        aws_access_key_id=MINIO_PARAM["key"],
        aws_secret_access_key=MINIO_PARAM["secret"],
    )
    MC = Minio(
        SETTING["minio"]["client_endpoint"].replace("http://", ""),
        access_key=MINIO_PARAM["key"],
        secret_key=MINIO_PARAM["secret"],
        secure=False,  # True if using HTTPS
    )
except KeyError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Remote service setup failed: {e}")

# PostgreSQL connection parameters (lazy loading)
POSTGRES_PARAM = {}
PS = None

# Initialize PostgreSQL parameters but don't connect yet
try:
    POSTGRES_PARAM = {
        "database": SETTING["postgres"]["database"],
        "user": SETTING["postgres"]["username"],
        "password": SETTING["postgres"]["password"],
        "host": SETTING["postgres"]["server_url"],
        "port": SETTING["postgres"]["port"],
    }
except KeyError as e:
    print(f"PostgreSQL configuration error: {e}")
    POSTGRES_PARAM = {}


def get_postgres_connection():
    """
    Get PostgreSQL connection with lazy initialization.

    Returns:
        psycopg2.connection: PostgreSQL connection object, or None if failed
    """
    global PS

    if PS is not None:
        # Check if connection is still alive
        try:
            PS.cursor().execute("SELECT 1")
            return PS
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            # Connection is dead, need to reconnect
            PS = None

    # Create new connection
    if POSTGRES_PARAM:
        try:
            PS = psycopg2.connect(**POSTGRES_PARAM)
            return PS
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}")
            PS = None
            return None
    else:
        print("PostgreSQL configuration not available")
        return None
