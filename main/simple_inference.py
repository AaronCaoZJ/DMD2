"""
简单的推理脚本，用于测试训练好的 SDv1.5 模型
"""
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch
import argparse
import os


def load_generator(checkpoint_path):
    """加载生成器模型"""
    print(f"Loading generator from {checkpoint_path}")
    
    # 加载基础模型结构
    generator = UNet2DConditionModel.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="unet"
    ).float()
    
    # 加载训练好的权重
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(state_dict, strict=True)
    generator.requires_grad_(False)
    
    return generator


@torch.no_grad()
def generate_images(generator, vae, text_encoder, tokenizer, prompts, args):
    """生成图像 - 使用4步backward simulation"""
    generator.eval()
    all_images = []
    
    # 设置4步去噪的时间步
    # 对于1000步的scheduler，4步去噪通常使用: 999, 749, 499, 249
    denoising_steps = [999, 749, 499, 249]  # 4步
    
    for idx, prompt in enumerate(prompts):
        print(f"Generating image {idx+1}/{len(prompts)}: {prompt}")
        
        # 为每张图片设置固定种子（基于全局种子 + 图片索引）
        torch.manual_seed(args.seed + idx)
        torch.cuda.manual_seed(args.seed + idx)
        
        # 编码文本
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(args.device)
        text_embedding = text_encoder(text_input_ids)[0]
        
        # 生成初始随机噪声
        noisy_image = torch.randn(
            1, 4, args.latent_resolution, args.latent_resolution,
            dtype=torch.float32,
            device=args.device
        )
        
        # 4步去噪过程
        for step_idx, timestep_value in enumerate(denoising_steps):
            timesteps = torch.ones(1, device=args.device, dtype=torch.long) * timestep_value
            
            # 使用生成器预测噪声
            predicted_noise = generator(
                noisy_image, timesteps, text_embedding
            ).sample
            
            # 从噪声预测中恢复x0（干净图像）
            # x0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
            # 使用alphas_cumprod计算
            alpha_t = 0.0047 if timestep_value == 999 else (1 - timestep_value/1000.0)
            alpha_t = torch.tensor([alpha_t], device=args.device).reshape(-1, 1, 1, 1)
            
            generated_image = (noisy_image - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            
            # 如果不是最后一步，添加噪声到下一个时间步
            if step_idx < len(denoising_steps) - 1:
                next_timestep = denoising_steps[step_idx + 1]
                # 简化版本：直接使用生成的图像作为下一步的输入，添加适当的噪声
                noise = torch.randn_like(generated_image)
                next_alpha = 0.0047 if next_timestep == 999 else (1 - next_timestep/1000.0)
                next_alpha = torch.tensor([next_alpha], device=args.device).reshape(-1, 1, 1, 1)
                noisy_image = next_alpha.sqrt() * generated_image + (1 - next_alpha).sqrt() * noise
            else:
                # 最后一步，使用生成的图像
                noisy_image = generated_image
        
        # 解码latents到图像
        generated_latents = noisy_image / 0.18215
        images = vae.decode(generated_latents).sample
        
        # 转换到[0, 255]
        images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        
        # 转换为PIL图像
        pil_image = Image.fromarray(images[0])
        all_images.append(pil_image)
    
    return all_images


def save_images(images, prompts, output_dir):
    """保存生成的图像"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image, prompt) in enumerate(zip(images, prompts)):
        # 创建安全的文件名
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
        safe_prompt = safe_prompt[:50]
        
        filename = f"{idx:04d}_{safe_prompt}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath)
        print(f"Saved: {filepath}")
    
    print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple inference for trained SDv1.5 model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint (pytorch_model.bin)")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Output directory")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=[
                            "a beautiful landscape with mountains",
                            "a cute cat sitting on a table",
                            "a futuristic city at night",
                            "a portrait of a woman",
                            "a cup of coffee on a wooden table"
                        ],
                        help="Prompts for generation")
    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # 设置全局随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 80)
    print("Simple Inference for SDv1.5")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Number of images: {len(args.prompts)}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # 加载模型组件
    print("\nLoading models...")
    generator = load_generator(args.checkpoint).to(args.device)
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae"
    ).to(args.device).float()
    vae.eval()
    
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder"
    ).to(args.device).float()
    text_encoder.eval()
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )
    
    # 生成图像
    print("\nGenerating images...")
    images = generate_images(generator, vae, text_encoder, tokenizer, args.prompts, args)
    
    # 保存图像
    print("\nSaving images...")
    save_images(images, args.prompts, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
