:boom: **Good news! Our new work exhibits state-of-the-art performances on the [DocUNet Benchmark](https://www3.cs.stonybrook.edu/~cvl/docunet.html) dataset:
[DocScanner: Robust Document Image Rectification with Progressive Learning](https://drive.google.com/file/d/1mmCUj90rHyuO1SmpLt361youh-07Y0sD/view?usp=share_link)** with [Repo](https://github.com/fh2019ustc/DocScanner).

:boom: **Good news! A comprehensive list of [Awesome Document Image Rectification](https://github.com/fh2019ustc/Awesome-Document-Image-Rectification) methods is available.** 

# DocTr++
The official code for “Deep Unrestricted Document Image Rectification”.
![Demo](assets/github_demo.png)


Any questions or discussions are welcomed!


## Training

- For geometric unwarping, we train the network using the [Doc3D](https://github.com/fh2019ustc/doc3D-dataset) dataset.


## Inference 
1. Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1eZRxnRVpf5iy3VJakJNTKWw5Zk9g-F_0?usp=sharing), and put them to `$ROOT/model_pretrained/`.
2. Put the distorted images in `$ROOT/distorted/`.
3. Geometric unwarping. The rectified images are saved in `$ROOT/rectified/` by default.
    ```
    python inference.py
    ```


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
@article{feng2021docscanner,
  title={Deep Unrestricted Document Image Rectification},
  author={Feng, Hao and Liu, Shaokai and Deng, Jiajun and Zhou, Wengang and Wu, Feng and Li, Houqiang},
  journal={arXiv},
  year={2023}
}
```

