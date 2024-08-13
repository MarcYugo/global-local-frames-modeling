## Joint Global-Local Frames Modeling to Enhance Semantic Alignment for Zero-shot Long Video Editing

|[ paper ]()|

### Environment Deployment

+ Anaconda environment
    
    ```bash
    conda env create -n glfm -f glfm_environment.yml
    conda activate glfm 
    ```

+ Pre-trained models
    ```bash
    python install.py --models_dir /your/path/to/models/
    ```
### Video Editing

You can use this shell scripts to edit your videos.
```bash
bash run_editing.sh
```

### Examples

+ Example 1

    Inversion prompt: A girl with blonde hair in a light blue T-shirt and white sports skirt is playing tennis

    Target prompt: A girl with black hair in a light purple T-shirt and gray sports skirt is playing tennis

    Total frames: 200, batch size for inversion: 100

    
    <table class="center">
    <tr>
    <td><video src="[./results/a_girl_play_tennis1_25fps.mp4](https://github.com/MarcYugo/global-local-frames-modeling/blob/main/results/a_girl_play_tennis1_25fps.mp4)" autoplay></td>
    </tr>
    </table>
    

+ Example 2

    Inversion prompt: A woman is running

    Target prompt: Black widow is running

    Total frames: 400, batch size for inversion: 100

    <center>
    <video id="video" controls="" preload="none" poster="封面">
    <source id="mp4" src="./results/a_woman_running_20fps.mp4" type="video/mp4">
    </video>
    </center>

+ Example 3

    Inversion prompt: A man is running

    Target prompt: Bat man is running

    Total frames: 300, batch size for inversion: 100

    <center>
    <video id="video" controls="" preload="none" poster="封面">
    <source id="mp4" src="./results/a_man_running_25fps.mp4" type="video/mp4">
    </video>
    </center>

+ Example 4

    Inversion prompt: There is thick snow on the roof of a house

    Target prompt: There is thick snow on the roof of a house, disney movie Frozen style

    Total frames: 300, batch size for inversion: 100

    <center>
    <video id="video" controls="" preload="none" poster="封面">
    <source id="mp4" src="./results/snow_house_20fps.mp4" type="video/mp4">
    </video>
    </center>

+ Example 4

    Inversion prompt: A woman is playing with a siberian husky. Tattooed arms.

    Target prompt: A woman is playing with a golden retriever. Normal-skin arms.

    Total frames: 400, batch size for inversion: 100

    <center>
    <video id="video" controls="" preload="none" poster="封面">
    <source id="mp4" src="./results/a_dog_play_20fps.mp4" type="video/mp4">
    </video>
    </center>

### Acknowledgment

Our repository is based on [STEM-SA](https://github.com/STEM-Inv/stem-inv). Thanks for their contribution to source code.
