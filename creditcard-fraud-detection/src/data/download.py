"""Dataset retrieval utilities.

In dev, the PaySim CSV is expected to be uploaded into MinIO (S3-compatible),
by default in bucket `data` with object key `paysim.csv`.

Airflow containers already expose the needed env vars via docker-compose:
- MINIO_ENDPOINT (e.g. http://minio:9000)
- MINIO_ACCESS_KEY / MINIO_SECRET_KEY (or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MinioLocation:
    bucket: str
    key: str


def _get_minio_endpoint() -> str:
    endpoint = os.getenv("MINIO_ENDPOINT") or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    if not endpoint:
        raise ValueError(
            "Missing MINIO_ENDPOINT (or MLFLOW_S3_ENDPOINT_URL). "
            "Airflow must be able to reach MinIO over the Docker network (e.g. http://minio:9000)."
        )
    return endpoint


def _get_minio_credentials() -> tuple[str, str]:
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            "Missing MinIO credentials (MINIO_ACCESS_KEY/MINIO_SECRET_KEY or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)."
        )
    return access_key, secret_key


def get_s3_client():
    """Return an S3 client configured for MinIO."""
    endpoint_url = _get_minio_endpoint()
    access_key, secret_key = _get_minio_credentials()
    region = os.getenv("AWS_REGION", "us-east-1")

    # Path-style addressing is the most compatible for MinIO.
    config = Config(signature_version="s3v4", s3={"addressing_style": "path"})
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=config,
    )


def download_dataset(
    output_path: str,
    location: MinioLocation | None = None,
    overwrite: bool = False,
) -> str:
    """Download PaySim CSV from MinIO into a local path.

    Args:
        output_path: Destination path inside the container (e.g. /tmp/data/raw/paysim.csv)
        location: MinIO bucket/key (defaults to env vars or bucket `data` + key `paysim.csv`)
        overwrite: If False and file exists, do nothing.

    Returns:
        output_path
    """
    if not overwrite and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("Dataset already present at %s (skipping download)", output_path)
        return output_path

    if location is None:
        bucket = os.getenv("PAYSIM_BUCKET", "data")
        key = os.getenv("PAYSIM_OBJECT_KEY", "paysim.csv")
        location = MinioLocation(bucket=bucket, key=key)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    client = get_s3_client()
    logger.info("Downloading s3://%s/%s -> %s", location.bucket, location.key, output_path)

    try:
        client.head_object(Bucket=location.bucket, Key=location.key)
    except Exception as e:
        raise FileNotFoundError(
            f"MinIO object not found: s3://{location.bucket}/{location.key}. "
            "Upload the CSV in MinIO console (http://localhost:9001) or set PAYSIM_BUCKET/PAYSIM_OBJECT_KEY. "
            f"Original error: {e}"
        )

    client.download_file(location.bucket, location.key, output_path)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise IOError(f"Download succeeded but output file is missing/empty: {output_path}")

    return output_path
