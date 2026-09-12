import redis
from base.config import config as conf
from base.logger import logger

class RedisClientManagement:
    def __init__(
            self,
            host:str=conf.redis.host,
            port:int=conf.redis.port,
            password:str=conf.redis.password,
            db:int=conf.redis.db,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client = None


    def init(self):
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            decode_responses=True
        )
        self._check_client_status()


    def _check_client_status(self):
        try:
            status = self.client.ping()
            if status:
                logger.info("Redis 连接成功")
        except Exception as e:
            logger.error(f"Redis连接异常:{e}")
            raise redis.ConnectionError("Redis 连接失败")

    def client_close(self):
        self.client.close()


if __name__ == "__main__":
    client = RedisClientManagement()
    client.init()
















