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
