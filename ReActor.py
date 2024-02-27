import requests
import base64


def img2base64(img_path):
    # 这是一个图片转base64的函数
    with open(img_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read())
    return str(image_base64)[2:-1]


# 这个文件定义了服务器后端向stable Diffusion发送API请求的函数
def ReActor(face_base64):
    """
    :face_base64: 需要换的脸
    :return: 返回结果图片的base64字符串
    """
    # 请求体
    postdata = {
        "source_image": face_base64,
        "target_image": img2base64("result.jpeg"),
        "source_faces_index": [0],
        "face_index": [0],
        "upscaler": "4x_Struzan_300000",
        "scale": 2,
        "upscale_visibility": 1,
        "face_restorer": "CodeFormer",
        "restorer_visibility": 1,
        "restore_first": 1,
        "model": "inswapper_128.onnx",
        "gender_source": 0,
        "gender_target": 0,
        "save_to_file": 0,
        "result_file_path": "666.jpeg"
    }
    # 请求SD的API

    url = "http://127.0.0.1:7860/reactor/image"
    # 得到结果图片的base64字符串
    response = requests.post(url, json=postdata)
    response = str(response.json()["image"])
    return response
