from client_managed.mysql_client import MySQLClientManagement
from client_managed.redis_client import RedisClientManagement
from client_managed.milvus_client import  MilvusClientManagement
from service.qa_system import IntegratedQASystem
from base.logger import logger
import sys

def main():

    #数据库客户端初始化
    redis_client = RedisClientManagement()
    mysql_client = MySQLClientManagement()
    milvus_client = MilvusClientManagement()

    redis_client.init()
    mysql_client.init()
    milvus_client.init()

    #RAG系统初始化
    qa_system = IntegratedQASystem(redis_client=redis_client.client,
                                   mysql_client=mysql_client.client,
                                   milvus_client=milvus_client.client)

    try:
        print("\n欢迎使用问题系统")
        print(f"数据支持来源：{qa_system.conf.app.valid_sources}")
        print("输入查询进行问答,输入 exit 退出")
        choose_fun = int(input("是否更新RAG数据库 :\n 1:更新 0:不更新"))
        if choose_fun == 1:
            logger.info(f"main进入更新数据库流程")
            qa_system.document_load()
        print("\n")
        session_id = input("\n创建您的ID:").strip()
        while not session_id:
            session_id = input("\n创建您的ID:").strip()
        while True:
            query = input("\n请输入查询内容：").strip()
            if query.lower() in ["exit","quit","q"]:
                logger.info("退出系统")
                break
            source_filter = input(f"输入来源过滤({'/'.join(qa_system.conf.app.valid_sources)})(按Enter跳过)").strip()
            if source_filter and source_filter not in qa_system.conf.app.valid_sources:
                logger.warning(f"无效来源：{source_filter},忽略过滤")
                source_filter=None
            answer = qa_system.query(query,source_filter=source_filter,session_id=session_id)
            for value in answer:
                if value[1] == "False":
                    print(value[0],end="")
                else:
                    print(value[0])

    except Exception as e:
        logger.error(f"系统错误：{e}")

    except KeyboardInterrupt:
        sys.exit(0)

    except SystemExit:
        sys.exit(0)

    finally:
        # 数据库客户端关闭
        redis_client.client_close()
        mysql_client.client_close()
        milvus_client.client_close()

if __name__ == "__main__":
    main()













