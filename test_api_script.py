import os.path
import time
import requests
import cv2
from base64 import b64encode
import io, base64
from PIL import Image, PngImagePlugin
import socket, random ,json

url = "http://127.0.0.1:7861"

def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.png', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img

def send_json(client_socket, file_path):
    try:
        client_socket.sendall(bytes(file_path, encoding='utf-8'))
    except:
        print("Can't send data")
        print("Reconnecting...")
        time.sleep(10)
        client_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(SERVER_ADDR)
        print("Connected")

def prompt_generator(prompt, num=1):
    with open('C:\\Users\\chsjk\\PycharmProjects\\blacklist.txt') as f:
        black_lists = f.readlines()
    with open('C:\\Users\\chsjk\\PycharmProjects\\whitelist.txt') as f:
        white_list = ', '.join(f.readlines())
    file_path = os.path.join(os.getcwd(), "interrogate")
    # folders = os.listdir(file_path)
    folders = ['flavors_short.txt', 'mediums.txt', 'movements.txt']
    for folder in folders:
        text_f = os.path.join(file_path, folder)
        with open(text_f, 'r', encoding='utf-8') as f:
            if folder == 'flavors_short.txt':
                num+=1
            for _ in range(num):
                word = random.choice(f.read().split('\n'))
                while word in black_lists:
                    word = random.choice(f.read().split('\n'))
                prompt += ', '+word
    prompt+=white_list
    return prompt

def cmd_flags(url):
    json_msg = {
        "no_half": False,
        "reinstall_xformers": True,
        "xformers": False
    }
    requests.post(url=f'{url}/sdapi/v1/cmd-flags', json=json_msg, verify=False)


def sd_api_controlnet():
    script_args = [
                    3,
                   "Experts in South Korea have raised concerns that the environmental impact of the contaminated water release from Japan's Fukushima nuclear plant has not been properly assessed, with no investigation into the impact of radioactive substances on marine life or biological concentration considered",
                   'Record-breaking heatwave grips the regio, with temperatures soaring well above normal, leading to health concerns and increased strain on energy resources',
                   'An international team of astronomers, including researchers from the Korea Astronomy and Space Science Institute, has observed for the first time the accretion disk and powerful jet of the supermassive black hole at the center of the galaxy M87',
                   'Wildfires continue to ravage vast areas, fueled by prolonged drought conditions, strong winds, and high temperatures, resulting in the evacuation of residents and significant damage to ecosystems',
                   '',
                   'Severe thunderstorms sweep across the area, accompanied by large hail, damaging winds, and frequent lightning, resulting in power outages and property damage',
                   '',
                   '',
                   '',
                   '',
                   2.0, 10.0, True, 0.0, 'R-ESRGAN 4x+', 1, 0, 0.0, 0, 75, 0.0001, 2.0, False, True, True, True, True, 'Linear', 3
                ]

    json_msg = {
        "prompt": '',
        "negative_prompt": '',
        "seed": -1,
        "subseed": -1,
        "subseed_strength": 0,
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": 768,
        "height": 768,
        "restore_faces": False,
        "eta": 0,
        "sampler_index": "Euler a",
        "script_name": "interpolation2",
        "script_args": script_args,
    }

    ### Override
    sd_model_list = ['v2-aZovyaRPGArtistTools_sd21768V1.safetensors', 'v2-1_768-ema-pruned.ckpt', 'v2_kitchensink2fp16_.safetensors','v2-doubleExposurePhoto_doubleExposurePhoto.safetensors', 'v2-fkingScifiV2_v21f.safetensors', 'v2-illuminati.safetensors','v2-pixhell_v20.safetensors','v2-prmj_v1.safetensors']
    import random
    sd_model = random.choice(sd_model_list)
    override_settings = {}
    override_settings["sd_model_checkpoint"] = sd_model
    override_payload = {
        "override_settings": override_settings
    }

    json_msg.update(override_payload)


    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=json_msg, verify=False)
    js = response.json()
    image = Image.open(io.BytesIO(base64.b64decode(js["images"][0])))

    # information text
    png_payload = {
        "image": "data:image/png;base64," + js['images'][0]
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
    # pnginfo = PngImagePlugin.PngInfo()
    # pnginfo.add_text("parameters", response2.json().get("info"))

    # for i in js['images']:
    #     image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
    #
    #     png_payload = {
    #         "image": "data:image/png;base64," + i
    #     }
    #     response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
    #
    #     pnginfo = PngImagePlugin.PngInfo()
    #     pnginfo.add_text("parameters", response2.json().get("info"))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(os.getcwd(), 'api_result2', article.split('.')[0], str(idx))
    os.makedirs(model_path, exist_ok=True)

    path = os.path.join(model_path, f"{timestr}.png")
    image.save(path)

    with open(os.path.join(model_path, f"{timestr}.txt"), "w", encoding='utf-8') as f:
        f.write(response2.json().get("info"))


    return path

    # for i in r['images']:
    #     image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
    #
    #     png_payload = {
    #         "image": "data:image/png;base64," + i
    #     }
    #     response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)
    #
    #     pnginfo = PngImagePlugin.PngInfo()
    #     pnginfo.add_text("parameters", response2.json().get("info"))
    #     image.save('output.png', pnginfo=pnginfo)



if __name__ == "__main__":
    ip = "192.168.1.124"
    port = 8882
    SIZE = 1024
    SERVER_ADDR = (ip, port)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(SERVER_ADDR)
    print("Connected")

    ### receive prompt message from server ###
    while True:
        try:
            unity_msg = client_socket.recv(1024).decode('utf-8')
            if unity_msg:
                print(f"Received message: {unity_msg}")
                filepath = sd_api_controlnet()

            else:
                print("Connection Refused, Reconnecting...")
                time.sleep(5)
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((SERVER_ADDR))
                print("Connected")


        except Exception as e:
            print("exception error !! ")
            print(e)
            break





