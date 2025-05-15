
# Unity-Python实时视频流

# 实现效果
unity作为前端，可获取电脑摄像头的实时视频流图像，发送给python后端，python对于画面逐帧边缘处理，并返回给unity同步显示。
<img width="660" alt="image" src="https://github.com/user-attachments/assets/d164ebd7-b560-43f7-8062-ad5b44812171" />


## 配置Python后端

首先，确保你安装了必要的Python库：

```bash
pip install flask opencv-python numpy
```

### 创建一个Python脚本 `app.py`：

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def process_frame(frame):
    # 进行边缘检测处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_frame = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', processed_frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': encoded_frame})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8769)
```

---

## 设置Unity项目

### 1. 创建显示摄像头画面的脚本 `CameraDisplay.cs`

```csharp
using UnityEngine;

public class CameraDisplay : MonoBehaviour
{
    private WebCamTexture webcamTexture;

    void Start()
    {
        webcamTexture = new WebCamTexture();
        Renderer renderer = GetComponent<Renderer>();
        renderer.material.mainTexture = webcamTexture;
        webcamTexture.Play();
    }

    public WebCamTexture GetWebCamTexture()
    {
        return webcamTexture;
    }
}
```

---

### 2. 创建HTTP客户端脚本 `HttpClient.cs`

#### 安装 Newtonsoft.Json：

- 使用 Unity Package Manager 安装
- 打开 Unity：`Window -> Package Manager` → 左上角点击 `+` → `Add package from git URL...`
- 输入：

```text
https://github.com/jilleJr/Newtonsoft.Json-for-Unity.git#upm
```

然后点击 **Add**

#### 创建脚本：

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System;
using System.Text;
using Newtonsoft.Json.Linq;

public class HttpClient : MonoBehaviour
{
    public CameraDisplay cameraDisplay;
    public Renderer processedRenderer;
    private const string serverUrl = "http://127.0.0.1:8769/process";

    void Start()
    {
        StartCoroutine(SendFrames());
    }

    IEnumerator SendFrames()
    {
        while (true)
        {
            if (cameraDisplay.GetWebCamTexture() != null && cameraDisplay.GetWebCamTexture().isPlaying)
            {
                Texture2D tex = new Texture2D(cameraDisplay.GetWebCamTexture().width, cameraDisplay.GetWebCamTexture().height);
                tex.SetPixels(cameraDisplay.GetWebCamTexture().GetPixels());
                tex.Apply();
                byte[] bytes = tex.EncodeToJPG();
                string base64String = Convert.ToBase64String(bytes);

                yield return StartCoroutine(PostImage(base64String));
            }
            yield return new WaitForSeconds(0.1f); // 控制帧率
        }
    }

    IEnumerator PostImage(string base64String)
    {
        JObject json = new JObject { { "image", base64String } };
        byte[] postData = Encoding.UTF8.GetBytes(json.ToString());

        using (UnityWebRequest www = new UnityWebRequest(serverUrl, "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(postData);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string jsonResponse = www.downloadHandler.text;
                JObject responseJson = JObject.Parse(jsonResponse);
                string processedImageBase64 = responseJson["image"].ToString();
                byte[] imageBytes = Convert.FromBase64String(processedImageBase64);

                Texture2D tex = new Texture2D(2, 2);
                tex.LoadImage(imageBytes);
                processedRenderer.material.mainTexture = tex;
            }
            else
            {
                Debug.LogError("Error: " + www.error);
            }
        }
    }
}
```

---

### 3. 设置Unity场景

- 创建两个 Plane：
  - `CameraPlane`：附加 `CameraDisplay.cs`
  - `ProcessedPlane`：用于显示处理后画面

- 创建空对象 `HttpClientManager`，附加 `HttpClient.cs`
  - 在 `cameraDisplay` 字段拖入 `CameraPlane`
  - 在 `processedRenderer` 字段拖入 `ProcessedPlane` 的 Renderer 组件

---

## 🧪 运行和测试

1. 启动 Python 服务器：

```bash
python app.py
```

2. 运行 Unity 项目：
   - CameraPlane 显示摄像头原始画面
   - ProcessedPlane 显示处理后（边缘检测）结果
