# InsightFace's Face Recognition Service

The InsightFace's Face Recognition tool, as implemented and provided by [this repository](https://github.com/TreB1eN/InsightFace_Pytorch), wrapped as a CLAMS service. The model used is IR-SE50

This requires many modules including Torch and opencv. All required modules are listed in `requirements.txt`

Before running the program, the file that contains the face recognition model's weights must be downloaded from [IR-SE50 @ BaiduNetdisk](https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ) or from [IR-SE50 @ Onedrive](https://1drv.ms/u/s!AhMqVPD44cDOhkPsOU2S_HFpY9dC). Then the file must be renamed to `model_cpu_final.pth` and put inside the new folder named `save`. Finally, the `save` folder must be put inside the `work_space` folder.

The program face-recognizes one frame per second. For example, if the input video contains 30 frames per second, the frames that would be sent through the face recognition model are the 0th, 30th, 60th, 90th, ... frames, until the video ends.

## Using this service

First, make sure that the input video could be accessed by the program. In `input-mmif/example_1.json`, change the path under the "location" key to the absolute path of the video on your computer.

Use `python app.py -t input-mmif/example_1.json output-mmif/example_1.json` just to test the wrapping code without using a server. To test this using a server you run the app as a service in one terminal (when you add the optional  `--develop` parameter a Flask server will be used in development mode, otherwise you will get a production Gunicorn server):

```bash
$ python app.py [--develop]
```

And poke at it from another:

```bash
$ curl http://0.0.0.0:5000/
$ curl -H "Accept: application/json" -X POST -d@input-mmif/example_1.json "http://0.0.0.0:5000/?pretty=True" -o output-mmif/example_1.json
```

In CLAMS you usually run this in a Docker container. To create a Docker image

```bash
$ docker build -t clams-app-face .
```

Before running it as a container, make sure that the video could be located from inside the container. This could be done either by copying the video to the container (but this is not recommended, since it unnecessarily takes so much time and space), or, more effectively, by creating a Docker volume and mounting it with the folder that contains the video.

The most effective way to create a Docker volume and mount it at runtime is by running the following command:

```bash
$ docker run - -rm -d -p 5000:5000 -v /your/absolute/path/to/the/video:/opt/clams/data/videos clams-app-face
```

In the above command `/your/absolute/path/to/the/video` should be substituted by the absolute path to the video in your computer.

Then poke at it (note that this time the location path in the input mmif file must start with `file:///opt/clams/data/videos`, which is where the video folder is mounted):

```
$ curl -H "Accept: application/json" -X POST -d@input-mmif/example_clams_1.json "http://0.0.0.0:5000/?pretty=True" -o output-mmif/example_1.json
```

The face recognition code will run on each video document in the input MMIF file. Every video document must be in the top level `documents` property. The file `input-mmif/example_1.json` has one such video document, which looks as follows:

```json
{
  "@type": "http://mmif.clams.ai/0.4.0/vocabulary/VideoDocument",
  "properties": {
    "mime": "video",
    "id": "d1",
    "location": "file:///Users/jinny/Desktop/app_insightFace/videos/Clinton_Lehrer.mp4"
  }
}
```
