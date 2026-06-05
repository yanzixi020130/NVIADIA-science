import os
import sys
from pathlib import Path
import json
import uvicorn
import asyncio
import traceback

from dotenv import load_dotenv
load_dotenv(override=True)  # 先加载 .env，确保 MINIO 等变量生效

from alpha.team import Team
from alpha.schema import Message
from team_config import *

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from src.data_processing_module import router as data_processing_router

# 端口从 .env 读取，给个兜底默认
PORT = int(os.getenv("PORT", "1101"))



UPLOAD_DIR="upload"
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，也可以指定具体源列表如["http://example.com", "https://example.com"]
    allow_credentials=False,  # 如果你的API需要 cookies 或者认证信息，设置为True
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有头部
)
app.include_router(data_processing_router)
# 挂载静态文件夹，使其可通过 /static/ 访问
os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")
async def start_round(websocket: WebSocket,team,idea,n_round,user_name,taskid,file_metadata):
    team.run_project(idea)
    await websocket.send_text("【XXX 开始: xxxx】")
    while n_round > 0:
        
        n_round -= 1
        for single_role in team.env.roles.values():
            observe_result=single_role._observe()
            if observe_result:
                single_role._no_think()
                if single_role.is_human:
                    human_str_s=f'【{single_role._setting} 等待您 {single_role.rc.todo.desc}】'
                    await websocket.send_text(human_str_s)
                    if single_role.rc.todo.PROMPT_TEMPLATE is not None:
                        await websocket.send_text(single_role.rc.todo.PROMPT_TEMPLATE)
                    # 发送等待标志
                    await websocket.send_text("[Pending]")
                    user_input = await websocket.receive_text()
                    full_reply_content=user_input
                    human_str_end=f'【{single_role._setting} 已完成 {single_role.rc.todo.desc}】'
                    await websocket.send_text(human_str_end)
                else:
                    
                    # if not response:
                    #     await websocket.send_text("本轮不发言！")
                    #     full_reply_content="本轮不发言！"
                    # elif isinstance(response, str):
                    #     full_reply_content=response

                    s=f"【{single_role._setting} 在做 : {single_role.rc.todo.desc}】"
                    await websocket.send_text("[start]")
                    # await websocket.send_text(s)
                    try:
                        # 可能引发异常的代码
                        full_reply_content= await single_role.rc.todo.run(single_role.rc.history,websocket,user_name,taskid,file_metadata)
                    except Exception as e:
                        # 使用traceback.print_exc()来打印异常堆栈信息
                        traceback.print_exc()
                        print(f"代码出错，请查看日志: {e}")
                        # await websocket.send_text(f"代码出错，请查看日志: {e}")
                        break

                    await websocket.send_text("[end]")    
                    f=f'【{single_role._setting} 已经完成 : {single_role.rc.todo.desc}】'
                    # await websocket.send_text(f)
                

                msg = Message(content=full_reply_content, role=single_role.profile, cause_by=single_role.rc.todo, sent_from=single_role)
                single_role.rc.memory.add(msg)
                single_role._set_state(state=-1)
                # Reset the next action to be taken.
                single_role.set_todo(None)
                # Send the response message to the Environment object to have it relay the message to the subscribers.
                single_role.publish_message(msg)
                break
    await websocket.send_text("【XXX 已完成: xxxx】")

@app.websocket("/start")
async def websocket_endpoint(websocket: WebSocket):
    
    team = Team()
    team.hire(
        [
            data_analysis_Input_Analyst(),
        ]
    )
    await websocket.accept()
    try:
        # 接收初始化数据
        init_data = await websocket.receive_json()
        #print(init_data)
        idea=init_data["idea"]
        n_round=int(len(team.env.roles))
        taskid=init_data["taskid"]
        user_name=init_data["user_name"]
        file_metadata=list(init_data["file_metadata"])

        await start_round(websocket,team,idea,n_round,user_name,taskid,file_metadata)
    except WebSocketDisconnect:
        # 在这里可以添加当客户端断开连接时的处理逻辑
        print("客户端已经主动断开连接！")
    except Exception as e:
        exception_type, exception_value, exception_traceback = sys.exc_info()
        print(f"Exception Type: {exception_type.__name__}")
        print(f"Exception Message: {exception_value}")
    try:
        await websocket.close()
        print("正常关闭连接！")
    except Exception as e:
        print("非正常关闭连接！")

@app.get("/roles")
async def get_teams():
    team = Team()
    team.hire(
        [
            data_analysis_Input_Analyst()
        ]
    )
    return team.env.get_roles()


@app.post("/uploadFile")
async def upload_file(
                      files: list[UploadFile] = File(...),
                      taskid: str = Form(...), 
                      ):
    
    """
    文件上传接口，增加了文件类型和大小的验证
    """

    # 确保上传目录存在
    SAVE_DIR=os.path.join(UPLOAD_DIR,taskid)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    try:
        # 将上传的文件保存到服务器
        for file in files:
            # 构建保存文件的完整路径
            file_path = os.path.join(SAVE_DIR, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
        upload_results=",".join([file.filename for file in files])
        
        return JSONResponse(content=f"文件 {upload_results} 上传成功", status_code=200)
    
    except HTTPException as http_exc:
        # 如果是已知的HTTP异常，直接抛出
        raise http_exc
    
    except Exception as e:
        # 处理其他异常情况
        return JSONResponse(content={"error": str(e)}, status_code=500)
 

@app.post("/files")
def list_files(taskid: str = Form(...)):
    # 构建文件夹路径
    folder_path = os.path.join("upload",taskid)
    
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 获取文件夹中的文件列表
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 返回JSON格式的结果
    return {"files": files}



if __name__=="__main__":
   # uvicorn.run(app='main:app', host="0.0.0.0", port=int(PORT))
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))
