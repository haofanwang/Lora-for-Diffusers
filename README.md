# Easy-Lora-Handbook

This repository provides the simplest tutorial code for AIGC researchers to use Lora in just a few lines. Using this handbook, you can easily play with any Lora model from active communities such as [Huggingface](https://huggingface.co/) and [cititai](https://civitai.com/).

Now, we also support [ControlNet-for-Diffusers](https://github.com/haofanwang/ControlNet-for-Diffusers).

# Background
## What is Lora?
Low-Rank Adaptation of Large Language Models ([LoRA](https://github.com/microsoft/LoRA)) is developed by Microsoft to reduce the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights. Lora attemptes to fine-tune the "residual" of the model instead of the entire model: i.e., train the $\Delta W$ instead of $W$.

$$
W' = W + \Delta W
$$

Where $\Delta W$ can be further decomposed into low-rank matrices : $\Delta W = A B^T $, where $A, \in \mathbb{R}^{n \times d}, B \in \mathbb{R}^{m \times d}, d << n$.
This is the key idea of LoRA. We can then fine-tune $A$ and $B$ instead of $W$. In the end, you get an insanely small model as $A$ and $B$ are much smaller than $W$.

This training trick is quite useful for fune-tuning customized models on a large general base model. Various text to image models have been developed built on the top of the official [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1). Now, with Lora, you can efficiently train your own model with much less resources.

## What is Safetensors?
Safetensors is a new simple format for storing tensors safely (as opposed to pickle) released by Hugging Face and that is still fast (zero-copy). For its efficiency, many stable diffusion models, especially Lora models are released in safetensors format. You can find more its advantages from [huggingface/safetensors](https://github.com/huggingface/safetensors) and install it via pip install.

```bash
pip install safetensors
```

# How to load Lora weights?

In this tutorial, we show to load or insert pre-trained Lora into diffusers framework. Many interesting projects can be found in [Huggingface](https://huggingface.co/) and [cititai](https://civitai.com/), but mostly in [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) framework, which is not convenient for advanced developers. We highly motivated by [cloneofsimo/lora](https://github.com/cloneofsimo/lora) about loading, merging, and interpolating trained LORAs. We mainly discuss models in safetensors format which is not well compatible with diffusers.

### Full model

A full model includes all modules needed (base model with or without Lora layers), they are usually stored in .ckpt or .safetensors format. We provide two examples below to show you how to use on hand.

- [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main) from Huggingface. 

- [dreamshaper](https://civitai.com/models/4384/dreamshaper) from Civitai. 

You can download .ckpt or .safetensors file only. Although diffusers does not support loading them directly, they do provide the converting script. First download diffusers to local.

```bash
git clone https://github.com/huggingface/diffusers
```

```bash
cd ./diffusers

# assume you have downloaded xxx.safetensors, it will out save_dir in diffusers format.
python ./scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path xxx.safetensors  --dump_path save_dir --from_safetensors

# assume you have downloaded xxx.ckpt, it will out save_dir in diffusers format.
python ./scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path xxx.ckpt  --dump_path save_dir
```

Then, you can load the model
```bash
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained(save_dir,torch_dtype=torch.float32)
```

### Lora model only
For now, diffusers cannot support load weights in Lora (usually in .safetensor format) . Here we show our attempts in an inelegant style. We also provide one example.

- [one-piece-wano-saga-style-lora](https://civitai.com/models/4219/one-piece-wano-saga-style-lora)

Note that the size of file is much smaller than full model, as it only contains extra Lora weights. In the case, we have to load the base model. It is also fine to just load stable-diffusion 1.5 as base, but to get satisfied results, it is recommanded to download suggested base model.

Our method is very straightforward: take out weight from .safetensor, and merge lora weight into a diffusers supported weight. We don't convert .safetensor into other format, we update the weight of base model instead.

Our script should work fine with most of models from [Huggingface](https://huggingface.co/) and [cititai](https://civitai.com/), if not, you can also modify the code on your own. Believe me, it is really simple and you can make it.

```bash
python convert_lora_safetensor_to_diffusers.py
```

We have made a PR for diffusers on this [issue](https://github.com/huggingface/diffusers/pull/2403), you can use it on diffusers soon!

# How to train your Lora?

Diffusers has provide a simple [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) to train your on Lora model. Please follow its instruction to install requirements.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-pokemon-model-lora" \
  --validation_prompt="cute dragon creature" --report_to="wandb"
```

Once you have trained a model using above command, the inference can be done simply using the StableDiffusionPipeline after loading the trained LoRA weights. You need to pass the output_dir for loading the LoRA weights which, in this case, is sd-pokemon-model-lora.

```bash
import torch
from diffusers import StableDiffusionPipeline

model_path = "your_path/sd-model-finetuned-lora-t4"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
```

## Train LoRA with ColossalAI framework

[ColossalAI](https://github.com/hpcaitech/ColossalAI) supports LoRA already. We only need modify a few lines on the top of [train_dreambooth_colossalai.py](https://github.com/huggingface/diffusers/blob/main/examples/research_projects/colossalai/train_dreambooth_colossalai.py). This example is for dreambooth, but you can easily adopt it regular text to image training. The generated LoRA weights are only for attention layers in UNet. If you want to support text encoder too, please use acceletate framework in diffusers, as ColossalAI does not support multiple models yet.

```bash
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor

# attention here! It is necessaray to init unet under ColoInitContext, not just lora layers
with ColoInitContext(device=get_current_device()):
        
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet", 
        revision=args.revision, 
        low_cpu_mem_usage=False
    )
    unet.requires_grad_(False)

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)

# DDP
unet = gemini_zero_dpp(unet, args.placement)

# config optimizer for colossalai zero, set initial_scale to large value to avoid underflow
optimizer = GeminiAdamOptimizer(unet, 
                                lr=args.learning_rate, 
                                betas=(args.adam_beta1, args.adam_beta2),
                                weight_decay=args.adam_weight_decay,
                                eps=args.adam_epsilon,
                                initial_scale=2**16, 
                                clipping_norm=args.max_grad_norm)
```

Here we go, the only thing is the initialization way of UNet. To save LoRA weights only,

```bash
torch_unet = get_static_torch_model(unet)
if gpc.get_local_rank(ParallelMode.DATA) == 0:
    torch_unet = torch_unet.to(torch.float32)
    torch_unet.save_attn_procs(save_path)
```

Then, do inference

```bash
from diffusers import StableDiffusionPipeline
import torch

model_path = "sd-model-finetuned-lora"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "A pokemon with green eyes and red legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")
```

You may find that the generated LoRA weight is only about 3MB size, this is because of the default setting. To increase the size, you can manually set the rank (dimension for low rank decomposition) for LoRA layers.

```bash
lora_attn_procs[name] = LoRACrossAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=128)
```

Then, the LoRA weights will be about 100-200MB size.
