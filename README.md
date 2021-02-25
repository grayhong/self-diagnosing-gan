# Self-Diagnosing GAN
Code for Self-Diagnosing GAN: Diagnosing Underrepresented Samples in Generative Adversarial Networks (https://arxiv.org/abs/2102.12033)

## Setup
This setting requires CUDA 11.
However, you can still use your own environment by installing requirements including PyTorch and Torchvision.

1. Install conda environment and activate it
```
conda env create -f environment.yml
conda activate torchenv
```

2. Install local package
```
pip install -e diagan-pkg
```

## Train for CIFAR-10 & CelebA
### Phase 1
1. Original GAN training
```
python train_mimicry_phase1.py --exp_name [exp name] --dataset [dataset] --root [dataset root] --loss_type [loss type] --seed [seed] --model [model]  --gpu [gpu id]  --save_logit_after [logit record start step] --stop_save_logit_after [logit record stop step]
```

### Example command 
To compare with original GAN, we train for 50k (75k) steps in total while our method uses checkpoint at 40k (60k) steps in phase 2, for CIFAR-10 (CelebA).

* CIFAR-10
```
python train_mimicry_phase1.py --exp_name cifar10-phase1 --dataset cifar10 --root ./dataset/cifar10 --loss_type ns --seed 1 --model sngan  --gpu 0  --save_logit_after 35000 --stop_save_logit_after 40000
```

* CelebA
```
python train_mimicry_phase1.py --exp_name celeba-phase1 --dataset celeba --root ./dataset/celeba --loss_type ns --seed 1 --model sngan  --gpu 0  --save_logit_after 55000 --stop_save_logit_after 60000
```

Downloading CelebA dataset might took very long. We recommend direct downloading from this [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

2. Top-k training
```
python train_mimicry_phase1.py --exp_name celeba-topk --dataset celeba --root ./dataset/celeba --loss_type ns --seed 1 --model sngan  --gpu 0  --save_logit_after 55000 --stop_save_logit_after 60000 --topk
```


### Phase 2
1. Dia-GAN (Ours)
* CIFAR-10
```
python train_mimicry_phase2.py --gpu 0 --exp_name cifar10-phase2 --resample_score ldr_conf_0.3_ratio_50_beta_1.0 --baseline_exp_name cifar10-phase1 --seed 1 --p1_step 40000 --dataset celeba --root ./dataset/cifar10  --loss_type ns  --num_steps 50000 --model sngan
```

* CelebA
```
python train_mimicry_phase2.py --gpu 0 --exp_name celeba-phase2 --resample_score ldr_conf_5.0_ratio_50_beta_1.0 --baseline_exp_name celeba-phase1 --seed 1 --p1_step 60000 --dataset celeba --root ./dataset/celeba  --loss_type ns  --num_steps 75000 --model sngan
```

2. GOLD Reweight
```
python train_mimicry_phase2.py --gpu 0 --exp_name celeba-gold --baseline_exp_name celeba-phase1 --seed 1 --p1_step 60000 --dataset celeba --root ./dataset/celeba  --loss_type ns  --num_steps 75000 --model sngan --gold
```

## Eval
1. Original GAN, GOLD, Top-k

Without DRS
```
python eval_gan.py --gpu 0 --exp_name celeba-phase1 --loss_type ns --netG_ckpt_step 75000 --dataset celeba --seed 1
```

With DRS
```
python eval_gan_drs.py --gpu 0 --exp_name celeba-phase1 --loss_type ns --netG_ckpt_step 75000 --dataset celeba --seed 1 --use_original_netD
```

2. Dia-GAN (Ours)
```
python eval_gan_drs.py --gpu 0 --exp_name celeba-phase1-diagan --loss_type ns --netG_ckpt_step 75000 --dataset celeba --seed 1
```

## Train for Colored-MNIST
### Phase 1
1. Original GAN training
```
python train_mimicry_color_mnist_phase1.py --gpu 0 --exp_name rd0.99-n10000-mnist_dcgan-bs64-loss_ns --model mnist_dcgan --major_ratio 0.99 --num_data 10000 --batch_size 64 --loss_type ns
```

2. PacGAN (Used packing of 2)
```
python train_mimicry_color_mnist_phase1.py --gpu 0 --exp_name rd0.99-n10000-mnist_dcgan-bs64-loss_ns-pack2 --model mnist_dcgan --major_ratio 0.99 --num_data 10000 --batch_size 64 --loss_type ns --num_pack 2
```

### Phase 2
1. Dia-GAN (Ours)
```
python train_mimicry_color_mnist_phase2.py --gpu 0 --exp_name rd0.99-mnist_dcgan-phase2 --baseline_exp_name rd0.99-n10000-mnist_dcgan-bs64-loss_ns --model mnist_dcgan --major_ratio 0.99 --p1_step 15000 --resample_score ldr_conf_1.0_ratio_50_beta_3.0 --batch_size 64 --loss_type ns --use_eval_logits 0
```

2. GOLD
```
python train_mimicry_color_mnist_phase2_gold.py --gpu 0 --exp_name rd0.99-mnist_dcgan-phase2-gold --baseline_exp_name rd0.99-n10000-mnist_dcgan-bs64-loss_ns --model mnist_dcgan --major_ratio 0.99 --p1_step 15000  --batch_size 64 --loss_type ns
```

## Eval for Colored-MNIST
We calculate Reconstruction Error (RE) score for Colored-MNIST.
1. Train CAE and calculate RE
```
python train_cae.py --exp_name rd0.99-mnist_dcgan-phase2 --netG_step 20000 --dataset color_mnist --model mnist_dcgan --root ./dataset/colour_mnist --num_data 10000 --major_ratio 0.99 --gpu 4 --loss_type ns
```

Then, we measure the difference of RE scores between baseline and our method for green samples.

2. Evaluation
```
python eval_ae_score_color_mnist.py --resample_exp_path ./exp_results/rd0.99-mnist_dcgan-phase2 --baseline_exp_path ./exp_results/rd0.99-n10000-mnist_dcgan-bs64-loss_ns --major_ratio 0.99
```
