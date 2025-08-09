import requests

# API 地址（根据你的服务运行地址修改）
API_URL = "http://localhost:8000/point_unsam"

# 要上传的图片路径（必须是 RGB 图）
IMAGE_PATH = "docs/demos/liver.png"
# 请求发送
with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(API_URL, files=files)

# 返回结果处理
if response.status_code == 200:
    result = response.json()
    print("✅ 推理成功！返回图像链接：")
    print("Marked input image:", result["marked_image"])
    print("Segmentation results:")
    for url in result["segmentation_results"]:
        print(" -", url)
else:
    print("❌ 请求失败：", response.status_code)
    print(response.text)
