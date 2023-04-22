import requests
import json
import numpy as np
import cv2
import base64
import time

from picamera import PiCamera
from picamera.array import PiRGBArray
from functools import partial
import RestfulAPI
import YanAPI

# 色卡检测的三个颜色
col_lis = ["blue", "green", "yellow"]


pic_lis = ["./mk1.jpg", "./mk2.jpg", "./mk3.jpg", "./mk4.jpg", "./mk5.jpg", "./mk6.jpg"]
# 6个截取区域的对应参数
crop_x = [50, 200, 20, 330, 20, 50]
crop_y = [36, 6, 35, 93, 16, 47]
crop_w = [156, 150, 139, 95, 26, 28]
crop_h = [200, 5, 20, 53, 68, 17]

# 六个区域对应头部舵机角度
angle1 = 80
angle2 = 90
angle3 = 104
angle_ls = 90
angleA = 80
angleB = 104
list_angle = [angle1, angle2, angle3, angle_ls, angleA, angleB]

# 定义五个颜色相关阈值
color_threshold = {
        "red":[[np.array([200, 0, 46]), np.array([179, 255, 255])]],
        "green":[[np.array([0, 100, 100]), np.array([200, 255, 255])]],
        "blue":[[np.array([20, 10, 100]), np.array([10, 255, 255])]],
        "yellow":[[np.array([20, 10, 20]), np.array([35, 21, 250])]],
        "purple":[[np.array([15, 50, 50]), np.array([150, 255, 205])]]}

# 读取图片
def getimage(imagename):
    with open(imagename, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    imgstr = img_byte.decode('ascii')
    return imgstr

# 机器人配置和摄像头初始化
def robot_init():
    global camera, rawCapture
    ip_add = "127.0.0.1"  # 调用本地服务
    YanAPI.yan_api_init(ip_add)
    YanAPI.set_robot_volume(50)

    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32  # 或者为40
    rawCapture = PiRGBArray(camera, size=(640, 480))

# 机器人说话，默认说话不可打断
def robot_say(words="",inter=False):
    return YanAPI.start_voice_tts(words, interrupt=inter)


# 机器人动作，默认一次且等待动作完成
def robot_motion(motion="", num=1, wait=True):
    res = YanAPI.start_play_motion(name=motion, repeat=num)
    if wait == True:
        waitdone()
    else:
        pass
    return res

# 飞机检测接口
def feiji(strff):
    url = "https://lab.qingsteam.cn/practice-web/practice/api/commonObjDetect?accessToken=PtG1Mw1ezc4EEhc7608Y4kc1vd1XK20v"   # 接口调用地址
    headers = {"Content-Type": "Application/json"}   # 请求头

    imgBase64 = strff
    # 调用getimage函数处理同一个根目录下的图片文件，得到base64编码数据

    requestBody = {   # 拼接JSON字符串
        "type":"commonObjDetect",
        "base64": "data:image/jpeg;base64,"+imgBase64
        }

    jsonBody = json.dumps(requestBody)   # 将JSON字符串转为JSON对象
    res = requests.post(url, jsonBody,headers=headers,verify=False).text   # 发送一个POST请求，并接收返回数据
    responseBody = json.loads(res)   # 将接收到的JSON格式数据转换为Python数据结构
    if responseBody['code'] == 1000:
        return False
    if responseBody['value'] == 0:
        return False
    for obja in responseBody["value"]["list"]:
        if obja['clsId'] == 4:
            print("飞机")
            return True
        if obja['clsId'] == 36: #这边的id == 36 在物体识别当中飞机会有概率识别成为滑板
            print("飞机")
            return True
    return False



# 飞机检测，loc为检测区域，aim为检测目标为有或者没有，typ为1时即直接检测，为其他值时为循环检测，循环3次自动跳出
def airplain(loc, aim=True, typ=1):
    global rawCapture, camera
    robot_motion("xy")
    loop = 0
    result = 0
    for capture in camera.capture_continuous(rawCapture,
                                             format="bgr",
                                             use_video_port=True):
        now_angle = list_angle[loc - 1]
        YanAPI.set_servos_angles({"NeckLR": now_angle})
        frame = capture.array
        cv2.imshow('1', frame)
        # 在准备下一帧时清除流
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            break
        x = crop_x[loc - 1]
        y = crop_y[loc - 1]
        w = crop_w[loc - 1]
        h = crop_h[loc - 1]
        img_crop = frame[(y + 5):(y + h + 5), (x + 5):(x + w + 5), ]  # 裁剪坐标为[y0:y1, x0:x1]


        cv2.imwrite(pic_lis[loc-1], img_crop)
        img = getimage(pic_lis[loc-1])
        result = feiji(img)
        if typ == 1:
            if result == aim:
                break
            else:
                pass
        else:
            if (result == aim) or (loop >= 3):
                break
            else:
                loop += 1
    return result

# 颜色识别，img输入检测图片，aim_color为目标颜色
def get_circles(img, aim_color):
    x = 0
    y = 0
    r = 0
    # 图形基本转换
    colors = color_threshold[aim_color]
    blurred = cv2.GaussianBlur(img, (11, 11), 0)  # 高斯模糊
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # 转换 hsv
    cnts = None

    for color in colors:
        # 根据阈值生成掩膜
        mask = cv2.inRange(hsv, color[0], color[1])
        # 形态学操作
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnt = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[-2]  # 检测颜色的轮廓
        if len(cnt) == 0:
            pass
        elif cnts is None:
            cnts = cnt[0]
        else:
            cnts = np.concatenate((cnts, cnt[0]), axis=0)
    if cnts is None:
        cnts = []
    else:
        cnts = [cnts]
    # 侦测到目标颜色
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
    return int(x), int(y), int(r), img


# 色卡判断,loc是目标区域的编号，5为A位置，6为B位置，mycolor为目标颜色
def color_check(loc, mycolor):
    loop = 0
    result = False
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        now_angle = list_angle[loc - 1]
        YanAPI.set_servos_angles({"NeckLR": now_angle})
        time.sleep(1)
        image = frame.array
        x, y, z, kl = get_circles(image, mycolor)
        rawCapture.truncate(0)
        if z > 15:
            cv2.circle(kl, (x, y), z, (0, 255, 0), 5)
        cv2.imshow('Color_tracking', kl)
        print("x=" + str(x) + ",y=" + str(y))
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            break

        if (x > crop_x[loc - 1]) and (x < (crop_x[loc - 1] + crop_w[loc - 1]) and y > crop_y[loc-1]):
            result = True
            break
        elif loop >= 3:
            result = False
            break
        else:
            loop += 1
    return result, loc, mycolor


def light_set(color, mode="breath", keep=5):
    YanAPI.set_robot_led(type="button", color=color, mode=mode)
    time.sleep(keep)

def is_feiji_remove(camera,rawCapture):
    # 侦测2号区域时身体姿态
    RestfulAPI.put_motions(name="xy")
    time.sleep(1)
    exist_flag = 0
    # 2号区域确认
    num = 2
    for capture in camera.capture_continuous(rawCapture,
                                                format="bgr",
                                                use_video_port=True):
        now_angle = list_angle[num - 1]
        RestfulAPI.put_servos_angles({"NeckLR": now_angle})
        time.sleep(1)
        frame = capture.array
        cv2.imshow('1', frame)
        # 在准备下一帧时清除流
        rawCapture.truncate(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
            break
        x = crop_x[num - 1]
        y = crop_y[num - 1]
        w = crop_w[num - 1]
        h = crop_h[num - 1]
        img_crop = frame[(y + 5):(y + h + 5), (x + 5):(x + w + 5), ]  # 裁剪坐标为[y0:y1, x0:x1]
        if num==2:
            cv2.imwrite("./mk2.jpg", img_crop)
            img_str = getimage("./mk2.jpg")
        else:
            cv2.imwrite("./mk.jpg", img_crop)
            img_str = getimage("./mk.jpg")
            
        if (feiji(img_str)==True):
            print("bad feiji is on Area 2 ...")
            exist_flag = 1
            time.sleep(1)
        else:
            if(exist_flag == 1):
                exist_flag = 0
                RestfulAPI.put_voice_tts("飞机已移除！", interrupt=False)
                print("bad feiji has be removed! ")
                break
    
#程序入口，上方均为函数设置
if __name__ == '__main__':
    #Yanshee摄像头设置
    camera = PiCamera()  #启动摄像头
    camera.resolution = (640, 480)  #摄像头分辨率为640*480
    camera.framerate = 32  # 摄像头帧率为32或40
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    RestfulAPI.put_devices_volume(100)#音量调节
    RestfulAPI.put_devices_led("button","red","blink")
    is_feiji_remove(camera,rawCapture)

    # 5/6号区域飞机检测
    num = 5
    while True:
        res = airplain(num, True, typ=2)
        if res:
            break
        elif num == 5:
            num = 6
        elif num == 6:
            num = 5
        else:
            pass
    if num == 5:
        robot_say("飞机位置在A")
        robot_say("向右转弯")
        robot_motion("right", 5)
    else:
        robot_say("飞机位置在B")
        robot_say("向左转弯")
        robot_motion("left", 5)

    # 四号区域检测
    airplain(4, True, 1)
    robot_say("向前直行")
    robot_motion("forward", 5)

    # 2号区域检测
    airplain(2, True, 1)
    robot_say("跑道转移已完成")

    # 紧急情况
    light_set(color="green", mode="blink")
    robot_say("紧急情况")
    robot_say("发生火灾！")
    robot_motion("hq", 3)
    RestfulAPI.put_devices_led("button","red","blink")
    time.sleep(4)
    
    # 跑道选择
    robot_say("向左转弯")
    robot_motion("left", 5)
    robot_say("向前直行")
    robot_motion("forward", 5)
    airplain(1, True, 1)
    robot_say("正常停止")
    robot_motion("stop")
    robot_say("跑道选择已完成")


    # 5/6区域色卡检测
    robot_motion("xy")
    light_set(color="blue", mode="breath")
    num = 5
    loop = 0
    while True:
        hav, area, colour = color_check(num, col_lis[loop])
        if hav:
            break
        else:
            loop += 1
            if loop > 2:
                loop = 0
                if num == 5:
                    num = 6
                else:
                    num = 5

    if num == 5:
        robot_say("颜色在A位置")
    if num == 6:
        robot_say("颜色在B位置")

    if colour == "blue":
        robot_say("飞机准备起飞")
    if colour == "green":
        robot_say("飞机起飞")
    if colour == "red":
        robot_say("天气原因，延误起飞")
    time.sleep(1)
    
    # 换岗休息
    robot_say("工作完成")
    RestfulAPI.put_motions(name="stand_up")
    time.sleep(3)
    #######向休息区移动######
    RestfulAPI.put_motions(name="reset")
    time.sleep(7)
    #右转
    RestfulAPI.put_motions(name="r22", repeat=6)
    #res = RestfulAPI.get_motions()
    # print(res["data"]["status"])
    time.sleep(1)
    while True:
        res = RestfulAPI.get_motions()
        if res["data"]["status"] == "idle" :
            break
    time.sleep(3)
    #直走
    RestfulAPI.put_motions(name="zhizou", repeat=10)
    time.sleep(10)
    RestfulAPI.put_motions(name="reset")
    time.sleep(0.2)
    # a = 70
    # res = RestfulAPI.put_servos_angles(
    #     {"RightAnkleFB": 90 + a, "LeftAnkleFB": 90 - a, "RightKneeFlex": 90 - a, "LeftKneeFlex": 90 + a}, 1000)
    RestfulAPI.put_motions(name="squat_down")
    time.sleep(3)

     



cv2.destroyAllWindows()

