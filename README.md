# 🧠 Conditional Diffusion for Liver Tumor Simulation & Inpainting (CT Scans)
### 📘 Course Project – **IE643: Deep Learning for Imaging Systems**

This repository contains our **IE643 course project**, where we implement a **Conditional Denoising Diffusion Probabilistic Model (DDPM)** for **realistic liver tumor synthesis and inpainting in CT scans** using:

- ✅ Healthy CT scans
- ✅ Liver segmentation masks
- ✅ Conditional diffusion modeling

The model learns to:
- Start from **pure Gaussian noise**
- Gradually **denoise into tumor-injected CT images**
- While being **conditioned on healthy anatomy + liver segmentation**

This enables:
- ✅ Synthetic tumor generation  
- ✅ Tumor inpainting  
- ✅ Data augmentation for medical AI  
- ✅ Robust training of segmentation & diagnosis models  

---

## 🖼️ Main Results (Input → Target → Generated Output)

<img width="3151" height="816" alt="diffusion_4panel_slice_433" src="https://github.com/user-attachments/assets/d335632a-eb7f-48f8-afb8-15dd49128818" />

---

## 📌 Key Features
- ✅ ε-parameterized DDPM training
- ✅ Conditional UNet architecture
- ✅ Forward + Reverse Diffusion Visualization
- ✅ Overfit Debug Mode
- ✅ Classifier-Free Guidance
- ✅ EMA (Exponential Moving Average)
- ✅ CT-aware Preprocessing Pipeline
- ✅ Mask-preserving segmentation workflow
- ✅ Full visualization of noising & denoising
  
Drive link for videos and presentation at different stages - https://drive.google.com/drive/folders/1yqMuM8hnUJt6Mk4-Gd_2J0DVcAuWSnUY?usp=sharing
---

## 🧬 Pipeline Overview

Healthy CT + Liver Mask
│
▼
Forward Diffusion (Add Noise)
│
▼
Conditional UNet (ε prediction)
│
▼
Reverse Diffusion (Denoising)
│
▼
Synthetic Tumor CT


---

## 📂 Project Structure

.
├── main2.py # Training + debugging + visualization

├── train.py # DDPM forward & reverse diffusion

├── models.py # Conditional UNet architecture

├── preprocess.py # CT preprocessing & normalization

├── visualise.py # Forward & reverse diffusion visualization

├── dataset.py # Dataset loader for CT NPZ volumes

├── inpainting.py # Tumor inpainting pipeline

├── utils.py # EMA, multi-GPU, checkpoints

├── processed_volumes/ # Preprocessed CT scans

├── processed_masks/ # Preprocessed segmentation masks

├── inpainted_volumes/ # Healthy inpainted volumes

├── checkpoints/ # Model checkpoints

└── visualizations/ # Training & sampling outputs

🏗️ Model Architecture

✅ Conditional UNet

✅ Time Embeddings

✅ Residual Blocks

✅ Multi-scale Encoder-Decoder

✅ Skip Connections

✅ GroupNorm + SiLU Activations

🎯 Project Objectives

✔️ Learn tumor appearance distribution

✔️ Simulate pathological CT scans

✔️ Improve segmentation robustness

✔️ Augment limited medical datasets

✔️ Enable tumor inpainting

⚠️ Current Limitations

Training stability still being optimized

High-resolution sampling is computationally expensive

Requires large-scale CT datasets for realistic generalization

🚀 Future Work

✅ DDIM Sampling for faster inference

✅ Multi-organ conditioning

✅ Multi-modal MRI + CT diffusion

✅ 3D volumetric diffusion

✅ Diffusion-based segmentation

🎓 Course Information

This project was developed as part of:

IE643 – Deep Learning for Imaging Systems
Indian Institute of Technology
Course Project on Medical Image Generation using Diffusion Models


