from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import StreamingResponse
from client_managed.mysql_client import MySQLClientManagement
from client_managed.redis_client import RedisClientManagement
from client_managed.milvus_client import  MilvusClientManagement
import json
import uuid
from main import IntegratedQASystem
from pydantic import BaseModel

app = FastAPI(title="问答系统",description="基于RAG + MySQL + Redis的问答系统 FastAPI接口")

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

class QueryInput(BaseModel):
    query:str
    source_filter:str
    session_id:str

@app.post("/query")
async def handle_query(request:QueryInput):
    try:
        query = request.query.strip()
        source_filter = request.source_filter
        session_id = request.session_id
    except Exception as e:
        raise HTTPException(status_code=400,detail="无效的JSON数据")

    if not query:
        raise HTTPException(status_code=400,detail="查询内容不能为空")

    if not session_id:
        session_id = str(uuid.uuid4())

    valid_source = qa_system.conf.app.valid_sources

    if source_filter and source_filter not in valid_source:
        raise HTTPException(status_code=400,detail=f"无效的学科类别。支持:{valid_source}")

    def generate_response():
        try:
            for token,is_complete in qa_system.query(query,source_filter=source_filter,session_id=session_id):
                message = {"response":token,"finish":is_complete,"ID":session_id}
                yield json.dumps(message,ensure_ascii=False)
        except Exception as e:
            error_msg = f"处理查询时发生错误:{str(e)}"
            qa_system.logger.error(error_msg)
            message = {"error":error_msg,"is_complete":True}
            yield f"data:{json.dumps(message,ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/even-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='localhost',port=8002)
























