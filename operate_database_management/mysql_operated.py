"""
作用：
    BM25知识库（jpkb表）
    对话历史（conversations表）
    MySQL:存储的是原始问题及问题答案

功能操作：
    创建数据库
    创建表
    插入数据
    获取问题
    获取答案
    关闭服务器
"""

from client_managed.mysql_client import MySQLClientManagement
from base.config import config as conf
from base.logger import logger
import pymysql
import pandas as pd

class OperateMySQL:
    def __init__(self,
                 client:MySQLClientManagement,
                 database:str=conf.mysql.database,
                 table_name:str=conf.mysql.table_name):
        self.client = client
        self.database = database
        self.table_name = table_name
        self._confirm_database()
        self._confirm_table()

    def _confirm_database(self):
        cursor = self.client.cursor()
        sql = """SELECT DATABASE()"""
        cursor.execute(sql)
        result = cursor.fetchone()
        if result[0]:
            logger.info(f"使用{result[0]}数据库")
        elif result[0] != self.database or result[0] is None:
            print(f"MySQL不存在{self.database}数据库,进入建库流程")
            self._create_database()
            sql = """SELECT DATABASE()"""
            cursor.execute(sql)
            result = cursor.fetchone()
            if result[0] is None:
                logger.error(f"MySQL：{self.database}数据库加载失败")
                raise pymysql.MySQLError("数据库加载失败")
            logger.info(f"使用{result[0]}数据库")

    def _confirm_table(self):
        cursor = self.client.cursor()
        sql = "SELECT DATABASE()"
        cursor.execute(sql)
        database = cursor.fetchone()
        if self.database == database[0]:
            sql = """SHOW TABLES"""
            cursor.execute(sql)
            result = cursor.fetchall()
            list_table = []
            for i in range(len(result)):
                list_table.append(result[i][0])
            if self.table_name not in list_table:
                print(f"MySQL数据库{self.database}:不存在{self.table_name},进入建表流程")
                self._create_table()
                sql = "SHOW TABLES"
                cursor.execute(sql)
                tables = cursor.fetchall()
                print("建表后检查列表:",tables)
            else:
                logger.info(f"MySQL数据库{self.database},存在{self.table_name}表")
        else:
            logger.error(f"MySQL未加载{self.database}数据库")


    def _create_database(self):
        try:
            cursor = self.client.cursor()
            sql = f"CREATE DATABASE IF NOT EXISTS {self.database}"
            cursor.execute(sql)
            self.client.commit()
            use_sql = f"USE {self.database}"
            cursor.execute(use_sql)
            print(f"MySQL创建数据库:{self.database}完成")
        except Exception as e:
            logger.error(f"MySQL创建数据库失败:{e}")


    def _create_table(self):
        try:
            self._confirm_database()
            sql = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name}(
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        subject_name VARCHAR(20),
                        question VARCHAR(1000),
                        answer VARCHAR(1000))
                    """
            cursor = self.client.cursor()
            cursor.execute(sql)
            self.client.commit()
            print(f"MySQL创建表{self.table_name}完成")
        except Exception as e:
            logger.error(f"MySQL创建表失败:{e}")


    def insert_data(self,file):
        try:
            data = pd.read_csv(file)
            cursor = self.client.cursor()
            for index,row in data.iterrows():
                sql = f"""INSERT INTO {self.table_name}(subject_name,question,answer) VALUES(%s,%s,%s)"""
                cursor.execute(sql,(row["学科名称"],row["问题"],row["答案"]))
                self.client.commit()
            logger.info("数据添加成功")
        except Exception as e:
            logger.error(f"MySQL插入数据失败:{e}")


    def get_question(self):
        try:
            cursor = self.client.cursor()
            sql = f"""SELECT question FROM {self.table_name}"""
            cursor.execute(sql)
            result = cursor.fetchall()
            if result:
                #print("MySQL获取问题表:",result)
                return result
        except Exception as e:
            logger.error(f"MySQL获取问题内容失败:{e}")


    def get_answer(self,query):
        try:
            cursor = self.client.cursor()
            sql = f"""SELECT answer FROM {self.table_name} WHERE question=%s"""
            cursor.execute(sql,(query,))
            result = cursor.fetchall()
            if result:
                #print("MySQL获取答案表:",result)
                return result
        except Exception as e:
            logger.error(f"MySQL获取答案内容失败:{e}")

    def close(self):
        self.client.close()

if __name__ == "__main__":
    from base.config import config as conf
    client = MySQLClientManagement()
    client.init()
    operator = OperateMySQL(client.client)
    operator.get_question()



























