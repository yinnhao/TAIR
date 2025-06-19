<div align="left">

# Text-Aware Image Restoration with Diffusion Models
<a href="https://arxiv.org/abs/2506.09993"><img src="https://img.shields.io/badge/arXiv-2506.09993-B31B1B"></a>
<a href="https://cvlab-kaist.github.io/TAIR/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>
<a href="https://huggingface.co/datasets/Min-Jaewon/SA-Text"><img src="https://img.shields.io/badge/HuggingFace-SA--Text-yellow?logo=huggingface&logoColor=yellow"></a>
<a href="https://huggingface.co/datasets/Min-Jaewon/Real-Text"><img src="https://img.shields.io/badge/HuggingFace-Real--Text-yellow?logo=huggingface&logoColor=yellow"></a>


[Jaewon&nbsp;Min<sup>1*</sup>](https://github.com/Min-Jaewon/) · 
[Jin&nbsp;Hyeon&nbsp;Kim<sup>2*</sup>](https://github.com/jinlovespho) · 
Paul&nbsp;Hyunbin&nbsp;Cho<sup>1</sup> · 
[Jaeeun&nbsp;Lee<sup>3</sup>](https://github.com/babywhale03) · 
Jihye&nbsp;Park<sup>4</sup> · 
Minkyu&nbsp;Park<sup>4</sup> · 
Sangpil&nbsp;Kim<sup>2†</sup> · 
Hyunhee&nbsp;Park<sup>4†</sup> · 
[Seungryong&nbsp;Kim<sup>1†</sup>](https://cvlab.kaist.ac.kr/)

<sup>1</sup> KAIST&nbsp;AI ·
<sup>2</sup> Korea&nbsp;University ·
<sup>3</sup> Yonsei&nbsp;University ·
<sup>4</sup> Samsung&nbsp;Electronics

<p align="center">
    <img src="assets/teaser.jpg">
</p>

<!-- <sub><sup>*</sup> Equal&nbsp;contribution  <sup>†</sup> Corresponding&nbsp;authors</sub> -->

<!-- ### [Paper&nbsp;(Coming&nbsp;soon)](#) | [Project&nbsp;Page](https://cvlab-kaist.github.io/TAIR) -->

</div>

## 📢 News
- 🤗 **2025.06.19** — **SA-Text** and **Real-Text** datasets are released along with the [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main)!
- 📄 **2025.06.12** — Arxiv paper is released! 
- 🚀 **2025.06.01** — Official launch of the repository and project page!


## 💾 SA-Text Dataset
**SA-Text** is a newly proposed dataset for **Text-Aware Image Restoration (TAIR)** task. It is built from  **SA-1B** dataset using our [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main) and  consists of **100K** image-text instance pairs with detailed scene-level annotations.

**Real-Text** is an evaluation dataset for real-world scenarios. It is constructed from [RealSR](https://github.com/csjcai/RealSR) and [DrealSR](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution) using same pipeline as above.

---

### Dataset Download

| Split             | Hugging Face 🤗 | Google Drive 📁 |
|------------------|:---------------:|:---------------:|
| **SA-Text**       | <div align="center">[Link](https://huggingface.co/datasets/Min-Jaewon/SA-Text)</div> | <div align="center">[Link](https://drive.google.com/file/d/1fJugZYInTIWUj0tY_iSddwTwmQDAoO-5/view?usp=sharing)</div> |
| **Real-Text**     | <div align="center">[Link](https://huggingface.co/datasets/Min-Jaewon/Real-Text)</div> | <div align="center">[Link](https://drive.google.com/file/d/1sIjeFe0Rq6IvYEC-pkz6aQ4ubuIge4xi/view?usp=sharing)</div> |


---

### Notes

- Each image is paired with one or more text instances with polygon-level annotations.
- The dataset follows a consistent annotation format, detailed in the [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main).
- We recommend using the dataset from Google Drive for testing our code.