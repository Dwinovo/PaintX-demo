import base64
import io
import time
import uvicorn
import sys
from sam_process import sam_process
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from img2img import img2img
from PIL import Image
from fastapi.responses import FileResponse
from ReActor import ReActor



# 定义这个请求体方便生成API
class ImageData(BaseModel):
    image_base64: str


app = FastAPI()

# CORS配置，我也不知道这是什么，但是必须要配置这个之后浏览器才能对我创建的API发送请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

width, height = None, None

res = None


def img2base64(img_path):
    # 这是一个图片转base64的函数
    with open(img_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read())
    return str(image_base64)[2:-1]


@app.post("/sam_image")
def sent_sam_image_data(image_base64: ImageData):
    """
    这是前端向后端发送原始图片并保存在后端函数
    后端固定将图片以“test.jpeg”存储
    """
    # 将base64转成图片
    image_data = base64.b64decode(image_base64.image_base64)

    # 使用Pillow的Image模块读取图片数据以获取宽度和高度
    image = Image.open(io.BytesIO(image_data))
    global width
    global height
    global res
    width, height = image.size

    # 保存图片
    with open("test.jpeg", 'wb') as f:
        f.write(image_data)

    # 获得SAM蒙版信息 segment anything model：自动抠图
    samProcess = sam_process("test.jpeg")
    res = samProcess.process()

    # 生成test.html文件，为什么要这样生成？因为我不太懂HTML，但是github上面有人做了一个和我们很像的模板，我爆改一半之后发现改不动了，就使用了这种“后端”控制“前端”的诡异方式
    # 具体可以阅读test.html文件（这个文件主要是生成了一个供用户选择遮罩信息的页面）
    return res


@app.post("/replaceFace")
def replaceFace(image_base64: ImageData):
    result = ReActor(image_base64.image_base64)
    print(result)
    return result


@app.post("/get_result")
def get_result(image_base64: ImageData):
    # 用户选择完图片的遮罩之后，点击“保存”按钮，会调用这个函数，此时我们已经拥有了原始图片和蒙版图片，根据这两个图片使用SD图生图重绘即可
    # 先把原始图片的base64编码存储到ordinary_base64中
    ordinary_base64 = img2base64("test.jpeg")

    # 再把蒙版图片的base64编码存储到mask_base64中
    mask_base64 = image_base64.image_base64

    # 通过img2img函数向SD发送请求
    result = img2img(ordinary_base64, mask_base64, width, height)
    print(result)
    # 将base64解码并存储在html/result.jpeg中
    result = base64.b64decode(result)
    with open("result.jpeg", "wb") as f:
        f.write(result)


@app.get("/ordinaryImage.jpeg")
def get_ordinary_image(version: int = Query(default=int(time.time()))):
    file_path = "test.jpeg"
    return FileResponse(file_path, media_type='image/jpeg')


@app.get("/resultImage.jpeg")
def get_result_image(version: int = Query(default=int(time.time()))):
    file_path = "result.jpeg"
    return FileResponse(file_path, media_type='image/jpeg')


@app.get("/get_res")
def get_result_image():
    return res


# 测试的一些代码，和正文无关
if __name__ == '__main__':
    uvicorn.run(app="main:app", host="127.0.0.1", port=8000)
