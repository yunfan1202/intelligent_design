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
