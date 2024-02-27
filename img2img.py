import requests


# 这个文件定义了服务器后端向stable Diffusion发送API请求的函数
def img2img(ordinary_base64, mask_base64, width, height):
    """
    :ordinary_base64:这个是指原始图片的base64编码
    :mask_base64: 这个是指经过用户选择遮罩之后的黑白遮罩图的base64编码
    :width: 图片的宽
    :height: 图片的高
    :return: 返回结果图片的base64字符串
    """
    # 请求体
    postdata = {
        "init_images": [ordinary_base64],
        "prompt": "masterpiece, best quality,perfect hand,five fingers,",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,wrong hand, bad hand,",
        "sampler_name": "Restart",
        "batch_size": 1,
        "steps": 20,
        "override_settings": {
            "sd_model_checkpoint": "majicmixRealistic_v7.safetensors [7c819b6d13]"
        },
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "denoising_strength": 0.75,
        "mask": mask_base64,
        "mask_blur": 4,
        "inpaint_full_res_padding": 32,
        "inpainting_fill": 1,
        "resize_mode": 0,
        "inpainting_mask_invert": 1,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "dw_openpose_full",
                        "model": "control_openpose-fp16 [9ca67cc5]",
                        "weight": 1,
                        "image": ordinary_base64,
                        "resize_mode": 1,
                        "processor_res": 512,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "control_mode": 2,
                    },
                    {
                        "enabled": True,
                        "module": "canny",
                        "model": "control_canny-fp16 [e3fe7712]",
                        "weight": 1,
                        "image": ordinary_base64,
                        "resize_mode": 1,
                        "processor_res": 512,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "control_mode": 2,
                    }
                ]
            },
            "ADetailer": {
                "args": [
                    True,
                    False,
                    {
                        "ad_model": "face_yolov8n.pt",

                        "ad_confidence": 0.3,
                        "ad_mask_k_largest": 0,
                        "ad_mask_min_ratio": 0.0,
                        "ad_mask_max_ratio": 1.0,
                        "ad_dilate_erode": 4,
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_blur": 4,
                        "ad_denoising_strength": 0.4,
                        "ad_inpaint_only_masked": True,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_use_inpaint_width_height": False,
                        "ad_inpaint_width": 512,
                        "ad_inpaint_height": 512,
                        "ad_use_steps": False,
                        "ad_steps": 28,
                        "ad_use_cfg_scale": False,
                        "ad_cfg_scale": 7.0,
                        "ad_use_checkpoint": False,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_use_vae": False,
                        "ad_vae": "Use same VAE",
                        "ad_use_sampler": False,
                        "ad_sampler": "DPM++ 2M Karras",
                        "ad_use_noise_multiplier": False,
                        "ad_noise_multiplier": 1.0,
                        "ad_use_clip_skip": False,
                        "ad_clip_skip": 1,
                        "ad_restore_face": False,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1.0,
                        "ad_controlnet_guidance_start": 0.0,
                        "ad_controlnet_guidance_end": 1.0
                    }
                ]
            }
        }
    }
    # 请求SD的API
    url = "http://127.0.0.1:7860/sdapi/v1/img2img"
    # 得到结果图片的base64字符串
    response = requests.post(url, json=postdata)
    response = str(response.json()["images"])[2:-2]
    return response
