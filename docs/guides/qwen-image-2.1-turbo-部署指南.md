# Qwen-Image-2.1-Turbo 部署指南（RTX 5080 16GB + 32GB 内存）

> **文档日期**：2026-11-08　**类型**：部署指南　**关联**：`docs/notes/20261108-qwen-image-2.1-turbo-调研.md`
> **适用硬件**：NVIDIA RTX 5080（16GB 显存）+ 32GB 系统内存；**推荐方案**：INT8 量化（画质接近全精度，余量充足）。
> 本指南提供两条独立部署路径，可任选其一，也可并存（共用模型文件不冲突）：
> - **路径 A：ComfyUI** —— 图形化工作流，日常出图首选，显存利用最优；
> - **路径 B：Diffusers（Python）** —— 代码调用，适合批量生产与集成进应用。

---

## 0. 前置条件

| 项目 | 要求 | 检查命令 |
|---|---|---|
| 显卡驱动 | 50 系 Blackwell 卡需较新驱动（550+，建议最新） | `nvidia-smi` |
| CUDA 运行时 | 由 PyTorch 自带，无需单独安装 Toolkit | —— |
| Python | 3.10 – 3.12 | `python --version` |
| Git | 任意较新版本 | `git --version` |
| 磁盘空间 | ≥ 25 GB（模型文件 + 缓存） | `df -h ~` |
| 网络 | 可访问 Hugging Face（或配置 HF 镜像） | —— |

⚠️ **RTX 50 系专用提醒**：5080 为 Blackwell 架构（sm_120），PyTorch 必须使用 **CUDA 12.8 构建**（torch ≥ 2.7 的 cu128 版本）。旧版 torch（cu118/cu121）无法在 5080 上正常运算。检查方法：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望：2.7.x / 2.7.x+cu128 类似版本，且 cuda.is_available() 为 True
```

国内网络下载模型前可配置 HF 镜像（写入 `~/.bashrc` 或每次会话执行）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 1. 路径 A：ComfyUI 部署（Step by Step）

### Step 0 —— 创建 conda 环境（推荐）

```bash
conda create -n comfyui python=3.11 -y
conda activate comfyui

# ⚠️ 必须先单独安装 cu128 构建（5080 是 Blackwell sm_120，PyPI 默认的 cu126 构建无法运行），
# 再执行 Step 1 的 requirements.txt 安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证：应输出 2.7.x+cu128 / 2.8.x+cu128 类似版本，且 True
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

> requirements.txt 中的 torch 未锁定具体构建，pip 会保留上面已装的 cu128 版本；
> 如发现被覆盖回其他构建，重跑上面的 torch 安装命令即可修复。

### Step 1 —— 安装或升级 ComfyUI 至 ≥ 0.39.0

Turbo 编辑模板要求 **ComfyUI ≥ 0.39.0**（0.37.0+ 仅支持基础编辑模板）。Qwen Image 2.1 相关节点（`Text Encode Qwen Image 2.1`、编辑子图、KV cache 节点）为**核心内置节点**，旧版本会显示红色缺失节点框，升级即可解决。

**情形 1：全新安装（git 方式，推荐）**
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

**情形 2：已有 git 安装，原地升级**
```bash
cd ComfyUI
git pull
pip install -r requirements.txt
```

**情形 3：Windows 便携版（portable）**
运行解压目录下的 `update/update_comfyui.bat` 即可。

**版本确认**：启动 ComfyUI 后在界面菜单查看版本号；或对 git 安装执行 `git describe --tags --always`。**必须 ≥ 0.39.0** 才能继续。

### Step 2 —— 下载模型文件（共 3 个）

官方 ComfyUI 权重仓库：**https://huggingface.co/Comfy-Org/Qwen-Image-2.1**（进入仓库 Files 页，Comfy-Org 系列文件通常位于 `split_files/` 目录下，按下面目标文件夹对号入座）：

| 文件 | 大小（约） | 放置目录 |
|---|---|---|
| `qwen_image_2.1_turbo_int8_convrot.safetensors` | 7.3 GB | `ComfyUI/models/diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors`（文本编码器） | 8–9 GB | `ComfyUI/models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | < 1 GB | `ComfyUI/models/vae/` |

浏览器直下或用 CLI（路径以仓库 Files 页实际显示为准）：
```bash
huggingface-cli download Comfy-Org/Qwen-Image-2.1 \
  --include "*turbo_int8*" "*int8_convrot*" "*vae*" \
  --local-dir ./ComfyUI-models
# 下载后按上表移入对应文件夹
```

**（可选）GGUF 路线**：若想尝试更小显存占用，从 **unsloth/Qwen-Image-2.1-GGUF** 下载 Q4_K_M（4.2GB）/ Q8_0（7.6GB）到 `models/unet/`，并安装 GGUF 加载节点：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/leejet/ComfyUI-GGUF   # 必须用 leejet 分支
pip install --upgrade gguf
```
⚠️ 注意：**city96 原版 ComfyUI-GGUF 不支持 2.1 架构**（报 `Unknown model architecture!`），同一环境只能装一个 GGUF 节点包，先删旧再装新。

### Step 3 —— 加载官方模板并跑通第一张图

1. 启动 ComfyUI：`python main.py`（默认 http://127.0.0.1:8188）；
2. 菜单 **Workflow → Browse Templates**，加载 **"Qwen Image 2.1 Turbo: Image Edit"**（8 步版；纯文生图可加载基础模板后换 Turbo 模型）；
3. 核对三个加载器：Turbo INT8 模型、`qwen3vl_8b_int8_convrot` 编码器、2.1 VAE；
4. 采样设置：**步数 8、CFG = 1、采样器 Euler**（Turbo 无需负向提示词；从基础工作流切换时这三项要同步改）；
5. 提示词输入 + 分辨率设置后运行，经 VAE 解码后在保存节点查看结果。

**分辨率规则**：边长须为 32 的倍数。测试阶段用 **1024×1024**；确认稳定后再上 2K 预设（2048×2048 方图、2752×1536 16:9）。超过 2K 训练尺寸后提示词遵循度会下降。

### Step 4 —— 进阶用法

- **RGBA 透明图**：透明通道需在**图像编辑模式**下通过提示词触发（如 "transparent background, clean alpha edges"），输出带 Alpha 的 PNG；
- **多参考图编辑**：`image_1` 是被编辑的主图；其余参考图在提示词中用 `<image2>`、`<image3>`… 引用（如"保持图1人物，添加图2的帽子"）；最多支持 10 张；
- **局部精准编辑**：在原图上画圈/涂抹标记区域，并在提示词中说明标记颜色含义；
- **速度参考**：ComfyUI 团队在 RTX 5090 上实测约 0.3 秒/步 @1024（3008×4000 大图约 6 秒/步）；5080 会略慢，**预热后 1024×1024 单张通常在 10–20 秒内**（含编码器/VAE 开销），以实测为准。

### Step 5 —— 验收清单

- [ ] ComfyUI 版本 ≥ 0.39.0，无红色缺失节点
- [ ] 1024×1024 文生图 8 步出图成功
- [ ] 2048×2048 出图成功，显存未溢出（`nvidia-smi` 观察）
- [ ] 编辑模板：给一张图"加蓝色围巾"类指令生效且未破坏其他区域
- [ ] 透明 RGBA 图输出，Alpha 边缘干净

### 常见问题

| 现象 | 原因与处理 |
|---|---|
| 红色缺失节点框 | ComfyUI 版本过旧 → 升级至 ≥ 0.39.0 |
| `Unknown model architecture!` | 装了 city96 原 GGUF 节点 → 换 leejet 分支 |
| 显存溢出（2K 时） | 降回 1024 测试；或改用 GGUF Q8_0/Q6_K；ComfyUI 自带显存分级卸载 |
| 出图仍走 40 步 | 用了基础模型而非 Turbo 检查点；或改了 `num_inference_steps`（对 Turbo 无效，调度固化在检查点内） |
| 画面噪点/异常 | CFG 未设为 1，或步数不是 8 |

---

## 2. 路径 B：Diffusers（Python）部署

### Step 1 —— 建立独立环境

**方式一：conda（推荐，与路径 A 的 comfyui 环境互不干扰）**

```bash
conda create -n qwen-image python=3.11 -y
conda activate qwen-image
```

**方式二：venv**

```bash
python -m venv qwen-image-env && source qwen-image-env/bin/activate   # Windows: qwen-image-env\Scripts\activate
```

> 两条路径各建独立环境（`comfyui` / `qwen-image`），依赖互不污染；
> 仓库中的性能实测脚本 `tools/bench_qwen_image.py` 后续在 `qwen-image` 环境中运行。

### Step 2 —— 安装依赖（顺序与版本敏感）

```bash
# 1) PyTorch：5080 必须 cu128 构建
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) 官方要求的基础版本
pip install "transformers>=5.17"

# 3) Diffusers 必须装源码版（须包含 PR #14950：pipeline 配置采样 sigma）
pip install git+https://github.com/huggingface/diffusers.git

# 4) 其余依赖
pip install accelerate pillow
```
⚠️ 官方 README 写的 `torch>=2.4.0` 是通用下限；**5080 实际必须 cu128（torch ≥ 2.7）**，否则 `torch.cuda.is_available()` 为 False 或算子报错。

### Step 3 —— 文生图（首次运行自动从 Hugging Face 下载权重，约占 30GB 缓存）

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1-Turbo",
    dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="A red panda sitting on a wooden bench in a quiet garden, "
           "soft morning light, detailed fur, natural photography.",
    width=1024,
    height=1024,
    use_kv_cache=True,                       # 8 步推理的核心开关
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("t2i_test.png")
```

### Step 4 —— 16GB 显存的关键优化

BF16 全量检查点约 14GB+（DiT 7B ≈ 14GB，Qwen3-VL 8B 编码器另有约 16GB 权重），**无法同时常驻 16GB 显存**。官方方案：

```python
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1-Turbo", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()   # 替代 .to("cuda")：组件按需上 GPU、用完即回内存
```
- 32GB 内存足够支撑该模式；
- 代价是速度下降（组件搬运开销），**追求速度用路径 A 的 INT8 量化版**；
- 也可设 `pipe.vae.enable_tiling()`（如可用）进一步压低解码峰值。

### Step 5 —— 图像编辑

```python
from PIL import Image
image = pipe(
    prompt="Add a small blue knitted scarf around the red panda's neck. "
           "Keep the red panda, wooden bench, garden and lighting unchanged.",
    image=Image.open("t2i_test.png").convert("RGBA"),
    width=1024,
    height=1024,
    output_resolution=1024,
    use_kv_cache=True,
).images[0]
image.save("edit_test.png")
```

### Step 6 —— 采样调度注意事项

- 检查点**自带推荐 8 步调度**，pipeline 自动使用，无需手动配置；
- 仅改 `num_inference_steps` **不会**覆盖内置调度；确要自定义须显式传 `sigmas` 参数（官方声明其他调度未经评估，不建议）；
- 默认 CFG=1，无需负向提示词。

### 验收清单

- [ ] `torch.cuda.is_available()` 为 True 且版本为 cu128
- [ ] 1024 文生图成功（首次含下载，约 30GB；后续秒级加载）
- [ ] `enable_model_cpu_offload()` 模式下 2K 出图成功
- [ ] 编辑指令生效

---

## 3. 两条路径对比与选型建议

| 维度 | ComfyUI（路径 A） | Diffusers（路径 B） |
|---|---|---|
| 上手难度 | 低（模板开箱即用） | 中（依赖版本敏感） |
| 16GB 下的速度 | **快**（INT8 常驻显存） | 较慢（BF16 + CPU offload 搬运开销） |
| 批量/集成开发 | 一般（借 API 化改造） | **好**（原生 Python，易进生产管线） |
| 工作流可视化/迭代 | **好** | 一般 |
| 显存弹性 | GGUF/INT8 多档可选 | BF16 + offload 为主 |

**建议**：日常出图与提示词迭代用 ComfyUI；确认要产品化后，再以 Diffusers 路线做批量与集成。

## 4. 参考来源

- 官方 README（安装/Diffusers 代码/CPU offload）：https://github.com/QwenLM/Qwen-Image-2.1
- 官方模型卡（编辑示例/调度说明）：https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo
- ComfyUI 文件与模板、各显存档位选型、GGUF 分支说明：https://freakyjolly.com/qwen-image-2-1-edit-comfyui-guide
- 量化文件清单与组件职责：https://blog.csdn.net/gitblog_00834/article/details/155547833
- ComfyUI 权重仓库：https://huggingface.co/Comfy-Org/Qwen-Image-2.1
