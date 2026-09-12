from pymilvus import MilvusClient
from base.config import config as conf
from base.logger import logger


class MilvusClientManagement:
    def __init__(
            self,
            host:str=conf.milvus.host,
            port:str=conf.milvus.port,
    ):
        self.host = host
        self.port = port
        self.client = None

    def _get_uri(self):
        return f"http://{self.host}:{self.port}"

    def init(self):
        uri = self._get_uri()
        self.client = MilvusClient(
            uri = uri,
        )
        self._check_status()

    def _check_status(self):
        ver = self.client.get_server_version()
        if ver:
            logger.info(f"已连接Milvus.Ver{ver}")
        else:
            logger.error("连接Milvus失败")

    def client_close(self):
        self.client.close()

if __name__ == "__main__":
    client = MilvusClientManagement()
    client.init()




