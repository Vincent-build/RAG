import pymysql
from base.config import config as conf
from base.logger import logger

class MySQLClientManagement:
    def __init__(
            self,
            host:str=conf.mysql.host,
            port:int=conf.mysql.port,
            user:str=conf.mysql.user,
            password:str=conf.mysql.password,
            database:str=conf.mysql.database,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.client = None

    def init(self):
        try:
            self.client = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
            )
            logger.info("已连接MySQL数据库")
        except Exception as e:
            if f"Unknown database" in str(e):
                self._check_database()
                try:
                    self.client = pymysql.connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        database=self.database,
                    )
                    logger.info("已连接MySQL数据库")
                except Exception as e:
                    try:
                        print("连接MySQL数据库失败，连接MySQL服务器")
                        self.client = pymysql.connect(
                            host=self.host,
                            port=self.port,
                            user=self.user,
                            password=self.password,
                        )
                    except Exception as e:
                        logger.error(pymysql.MySQLError("连接MySQL服务器失败"))
                        raise SystemExit(1)
            else:
                logger.error(f"MySQL数据库连接异常:{e}")
                raise

    def _check_database(self):
        if self.database:
            client = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
            )
            cursor = client.cursor()
            sql = f"CREATE DATABASE IF NOT EXISTS {self.database}"
            cursor.execute(sql)
            client.commit()
            print(f"已创建数据库{self.database} 进行重连...")
            client.close()
        else:
            print("config中未正确配置MySQL的database内容")

    def client_close(self):
        self.client.close()


if __name__ == "__main__":
    client = MySQLClientManagement()
    client.init()



















