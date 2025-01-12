![block](./images/teaseradd.gif)

# GaussianDreamerPro: Text to Manipulable 3D Gaussians with Highly Enhanced Quality
### [Project Page](https://taoranyi.com/gaussiandreamerpro/) | [arxiv Paper](https://arxiv.org/abs/2406.18462)

[GaussianDreamerPro: Text to Manipulable 3D Gaussians with Highly Enhanced Quality](https://taoranyi.com/gaussiandreamerpro/)  

[Taoran Yi](https://github.com/taoranyi)<sup>1</sup>,
[Jiemin Fang](https://jaminfong.cn/)<sup>2†</sup>, [Zanwei Zhou](https://github.com/Zanue)<sup>3</sup>, [Junjie Wang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=9Nw_mKAAAAAJ)<sup>2</sup>, [Guanjun Wu](https://guanjunwu.github.io/)<sup>1</sup>,  [Lingxi Xie](http://lingxixie.com/)<sup>2</sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN)<sup>2</sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1✉</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1†</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2</sup> 

<sup>1</sup>HUST &emsp;<sup>2</sup>Huawei Inc. &emsp; <sup>3</sup>AI Institute, SJTU &emsp; 

<sup>†</sup>Project lead.  <sup>✉</sup>Corresponding author. 

![block](./images/dance.gif)

Recently, 3D Gaussian splatting (3D-GS) has achieved great success in reconstructing and rendering real-world scenes. To transfer the high rendering quality to generation tasks, a series of research works attempt to generate 3D-Gaussian assets from text. However, the generated assets have not achieved the same quality as those in reconstruction tasks. We observe that Gaussians tend to grow without control as the generation process may cause indeterminacy. Aiming at highly enhancing the generation quality, we propose a novel framework named GaussianDreamerPro. The main idea is to bind Gaussians to reasonable geometry, which evolves over the whole generation process. Along different stages of our framework, both the geometry and appearance can be enriched progressively. The final output asset is constructed with 3D Gaussians bound to mesh, which shows significantly enhanced details and quality compared with previous methods. Notably, the generated asset can also be seamlessly integrated into downstream manipulation pipelines, e.g. animation, composition, and simulation etc., greatly promoting its potential in wide applications.

## 🦾 Updates
- 1/12/2025: Release the rough code.
- 6/26/2024: Initializing the project, code will come soon.

## 🚀 Get Started
**Installation**
Install [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [Shap-E](https://github.com/openai/shap-e#usage) as fellow:
```

conda create -n GaussianDreamerPro python==3.8
conda activate GaussianDreamerPro
# Install pytorch3d
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c iopath iopath
conda install -c fvcore -c conda-forge fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/download.html
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/diff-gaussian-rasterization_2dgs
pip install ./submodules/simple-knn
```
Download [finetuned Shap-E](https://huggingface.co/datasets/tiange/Cap3D/blob/main/misc/our_finetuned_models/shapE_finetuned_with_330kdata.pth) by Cap3D, and put it in `./load`

**Quickstart**

For Basic 3D Asset Generation
```
cd stage1
python train.py --opt './configs/temp.yaml' --prompt "a DSLR photo of a pair of tan cowboy boots, studio lighting, product photography" --initprompt  "cowboy boots"

# For 24G GPU
python train.py --opt './configs/lowarm.yaml' --prompt "a DSLR photo of a pair of tan cowboy boots, studio lighting, product photography" --initprompt  "cowboy boots" 
```
For Quality Enhancement 3D Asset Generation
```
cd stage2
python meshexport.py -c "path/to/stage1/output/prompt@2024xxx"
python trainrefine.py --prompt "a DSLR photo of a pair of tan cowboy boots, studio lighting, product photography" --coarse_mesh_path "path/to/stage1/output/prompt@2024xxx/coarse_mesh/xxx.ply"
```

## 📑 Citation
If you find this repository/work helpful in your research, welcome to cite the paper and give a ⭐.
Some source code of ours is borrowed from [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer) and [SuGaR](https://github.com/Anttwo/SuGaR). We sincerely appreciate the excellent works of these authors.
```
@article{GaussianDreamerPro,
    title={GaussianDreamerPro: Text to Manipulable 3D Gaussians with Highly Enhanced Quality},
    author={Yi, Taoran and Fang, Jiemin and Zhou, Zanwei and Wang, Junjie and Wu, Guanjun and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Wang, Xinggang and Tian, Qi},
    journal={arXiv:2406.18462},
    year={2024}
}

@inproceedings{yi2024gaussiandreamer,
  title={Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models},
  author={Yi, Taoran and Fang, Jiemin and Wang, Junjie and Wu, Guanjun and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6796--6807},
  year={2024}
}
```
