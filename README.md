🔥 **Good news! Our demo has already exceeded 20,000 calls!**

🔥 **Good news! Our work has been accepted by IEEE Transactions on Multimedia.**

🚀 **Exciting update! We have created a demo for our paper, showcasing the generic rectification capabilities of our method. [Check it out here!](https://doctrp.docscanner.top/)**

🔥 **Good news! Our new work exhibits state-of-the-art performances on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset:
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link)** with [Repo](https://github.com/fh2019ustc/DocScanner).

🔥 **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 

# DocTr++

<p>
    <a href='https://project.doctrp.top/' target="_blank"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2304.08796' target="_blank"><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://demo.doctrp.top/' target="_blank"><img src='https://img.shields.io/badge/Online-Demo-green'></a>
</p>
  

![Demo](assets/github_demo.png)
![Demo](assets/github_demo_v2.png)
> **[DocTr++: Deep Unrestricted Document Image Rectification](https://arxiv.org/abs/2304.08796)**

> DocTr++ is an enhanced version of the original [DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction](https://github.com/fh2019ustc/DocTr), aiming to rectify various distorted document images in the wild,
whether or not the document is fully present in the image.

Any questions or discussions are welcomed!


## 🚀 Demo [(Link)](https://demo.doctrp.top/)
1. Upload the distorted document image to be rectified in the left box.
2. Click the "Submit" button.
3. The rectified image will be displayed in the right box.
4. Our demo environment is based on a CPU infrastructure, and due to image transmission over the network, some display latency may be experienced.

[![Alt text](https://user-images.githubusercontent.com/50725551/232952015-15508ad6-e38c-475b-bf9e-91cb74bc5fea.png)](https://demo.doctrp.top/)



## Inference 
1. Put the pretrained model to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Run the script and the rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```

## Evaluation
- ***Image Metrics:***  We propose the metrics MS-SSIM-M and LD-M, different from that for [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset. We use Matlab 2019a. Please compare the scores according to your Matlab version. We provide our Matlab interface file at ```$ROOT/ssim_ld_eval.m```.
- ***OCR Metrics:*** The index of 70 document (70 images) in [UDIR test set](https://drive.google.com/drive/folders/15rknyt7XE2k6jrxaTc_n5dzXIdCukJLh?usp=share_link) used for our OCR evaluation is provided in ```$ROOT/ocr_eval.py```. 
The version of pytesseract is 0.3.8, and the version of [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/) in Windows is recent 5.0.1.20220118. 
Note that in different operating systems, the calculated performance has slight differences.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}
```

```
@article{feng2023doctrp,
  title={Deep Unrestricted Document Image Rectification},
  author={Feng, Hao and Liu, Shaokai and Deng, Jiajun and Zhou, Wengang and Li, Houqiang},
  journal={IEEE Transactions on Multimedia},
  year={2023}
}
```

## Contact
For commercial usage, please contact Hao Feng ([haof@mail.ustc.edu.cn](haof@mail.ustc.edu.cn)).
