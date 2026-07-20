"""Object storage interface. Backend is swappable (local disk for dev,
S3-compatible for RunPod network volume, Azure Blob for prod) — callers only
ever depend on the `ObjectStorage` interface, never on a concrete backend."""

import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from config import get_settings


class ObjectStorage(ABC):
    @abstractmethod
    def put(self, key: str, local_path: str) -> str:
        """Upload the file at local_path under key. Returns a storage URI."""

    @abstractmethod
    def get(self, key: str, local_path: str) -> str:
        """Download the object at key to local_path. Returns local_path."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def url_for(self, key: str) -> str:
        """Best-effort reference URI for the object (not necessarily
        pre-signed/public — callers needing a signed URL should not assume
        this is one)."""


class LocalFilesystemStorage(ObjectStorage):
    """Dev-only backend — stores objects under a local root directory.
    No GPU/cloud dependency, matches the Step 1 "zero cost, zero GPU" requirement."""

    def __init__(self, root: str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put(self, key: str, local_path: str) -> str:
        dest = self._path(key)
        shutil.copyfile(local_path, dest)
        return self.url_for(key)

    def get(self, key: str, local_path: str) -> str:
        src = self._path(key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()

    def url_for(self, key: str) -> str:
        return f"file://{self._path(key)}"


class S3CompatibleStorage(ObjectStorage):
    """Works against any S3-compatible endpoint: AWS S3, a RunPod network
    volume exposed over S3, MinIO, etc. — governed entirely by
    object_storage_endpoint_url."""

    def __init__(self, bucket: str, endpoint_url: str | None, access_key: str | None, secret_key: str | None):
        import boto3

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def put(self, key: str, local_path: str) -> str:
        self._client.upload_file(local_path, self._bucket, key)
        return self.url_for(key)

    def get(self, key: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self._bucket, key, local_path)
        return local_path

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def url_for(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"


class AzureBlobStorage(ObjectStorage):
    """Prod backend, matches AdStitch's existing Azure Blob usage conventions
    where relevant. Only imports azure-storage-blob when actually selected."""

    def __init__(self, bucket: str, connection_string: str | None):
        from azure.storage.blob import BlobServiceClient

        if not connection_string:
            raise ValueError("Azure Blob backend requires a connection string")
        self._container_name = bucket
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._container = self._service.get_container_client(bucket)

    def put(self, key: str, local_path: str) -> str:
        with open(local_path, "rb") as f:
            self._container.upload_blob(name=key, data=f, overwrite=True)
        return self.url_for(key)

    def get(self, key: str, local_path: str) -> str:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(self._container.download_blob(key).readall())
        return local_path

    def exists(self, key: str) -> bool:
        return self._container.get_blob_client(key).exists()

    def delete(self, key: str) -> None:
        self._container.delete_blob(key)

    def url_for(self, key: str) -> str:
        return f"{self._container.url}/{key}"


@lru_cache
def get_object_storage() -> ObjectStorage:
    settings = get_settings()
    backend = settings.object_storage_backend

    if backend == "local":
        return LocalFilesystemStorage(settings.object_storage_local_root)
    if backend in ("s3", "runpod_volume"):
        return S3CompatibleStorage(
            bucket=settings.object_storage_bucket,
            endpoint_url=settings.object_storage_endpoint_url,
            access_key=settings.object_storage_access_key,
            secret_key=settings.object_storage_secret_key,
        )
    if backend == "azure_blob":
        return AzureBlobStorage(
            bucket=settings.object_storage_bucket,
            connection_string=settings.object_storage_endpoint_url,
        )
    raise ValueError(f"Unknown object_storage_backend: {backend}")
