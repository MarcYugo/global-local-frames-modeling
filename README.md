## Joint Global-Local Frames Modeling to Enhance Semantic Alignment for Zero-shot Long Video Editing

|[ paper ](https://doi.org/10.1016/j.neucom.2025.130836)|

Zewen Yu, Pengchong Qiao, Jie Chen, Xiaoqin Zhang

This project's code is intended for submission to Neurocomputing.

### Environment Deployment

+ Anaconda environment
    
    ```bash
    conda env create -n glfm -f glfm_environment.yml
    conda activate glfm 
    ```

+ Download pre-trained models
    ```bash
    python install.py --models_dir /your/path/to/models/
    ```
    if it doesn't work, you also can download them from [here](https://drive.google.com/file/d/10ey4rqj52wB3NDphH32-tR6it-VLdTLM/view?usp=sharing).
### Video Editing

You can use this shell script to edit your videos.
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
    <td><video src="https://github.com/user-attachments/assets/1840b450-5266-4cca-983b-9f70ab2f4700" autoplay></td>
    </tr>
    </table>
    

+ Example 2

    Inversion prompt: A woman is running

    Target prompt: Black widow is running

    Total frames: 400, batch size for inversion: 100

    <table class="center">
    <tr>
    <td><video src="https://github.com/user-attachments/assets/be603199-dd3c-4b29-98e2-f4686e1214bf" autoplay></td>
    </tr>
    </table>

+ Example 3

    Inversion prompt: A man is running

    Target prompt: Bat man is running

    Total frames: 300, batch size for inversion: 100

    <table class="center">
    <tr>
    <td><video src="https://github.com/user-attachments/assets/625e69a4-6ac3-4a89-b0a3-26248ca891ca" autoplay></td>
    </tr>
    </table>

+ Example 4

    Inversion prompt: There is thick snow on the roof of a house

    Target prompt: There is thick snow on the roof of a house, disney movie Frozen style

    Total frames: 300, batch size for inversion: 100

    <table class="center">
    <tr>
    <td><video src="https://github.com/user-attachments/assets/1b475c98-06bd-4ab1-afaf-ec424f94fb78" autoplay></td>
    </tr>
    </table>

+ Example 5

    Inversion prompt: A woman is playing with a siberian husky. Tattooed arms.

    Target prompt: A woman is playing with a golden retriever. Normal-skin arms.

    Total frames: 400, batch size for inversion: 100

    <table class="center">
    <tr>
    <td><video src="https://github.com/user-attachments/assets/154a59fb-80b0-4045-9e86-880c6b0a84c3" autoplay></td>
    </tr>
    </table>

### Acknowledgment

Our repository is based on [STEM-SA](https://github.com/STEM-Inv/stem-inv). Thanks for their contribution to source code.
