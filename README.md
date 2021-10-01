# Self-Diagnosing GAN (NeurIPS 2021)
Code for Self-Diagnosing GAN: Diagnosing Underrepresented Samples in Generative Adversarial Networks (NeurIPS 2021)

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
python train_mimicry_phase2.py --gpu 0 --exp_name cifar10-phase2 --resample_score ldr_conf_0.3_ratio_50 --baseline_exp_name cifar10-phase1 --seed 1 --p1_step 40000 --dataset cifar10 --root ./dataset/cifar10  --loss_type ns  --num_steps 50000 --model sngan
```

* CelebA
```
python train_mimicry_phase2.py --gpu 0 --exp_name celeba-phase2 --resample_score ldr_conf_5.0_ratio_50 --baseline_exp_name celeba-phase1 --seed 1 --p1_step 60000 --dataset celeba --root ./dataset/celeba  --loss_type ns  --num_steps 75000 --model sngan
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
python train_mimicry_color_mnist_phase2.py --gpu 0 --exp_name rd0.99-mnist_dcgan-phase2 --baseline_exp_name rd0.99-n10000-mnist_dcgan-bs64-loss_ns --model mnist_dcgan --major_ratio 0.99 --p1_step 15000 --resample_score ldr_conf_1.0_ratio_50 --batch_size 64 --loss_type ns --use_eval_logits 0
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
python eval_ae_score.py --resample_exp_path ./exp_results/rd0.99-mnist_dcgan-phase2 --baseline_exp_path ./exp_results/rd0.99-n10000-mnist_dcgan-bs64-loss_ns --major_ratio 0.99
```

## Train - MNIST-FMNIST
1. Phase 1
```
python train_mimicry_mnist_fmnist_phase1.py --exp_name fmnist-0.9-dcgan-seed1-phase1 --loss_type ns --model mnist_dcgan --gpu 2 --seed 1 --major_ratio 0.9 --num_data 60000
```

2. Phase 2
```
python train_mimicry_mnist_fmnist_phase2.py --exp_name fmnist-0.9-dcgan-seed1-phase2  --baseline_exp_name fmnist-0.9-dcgan-seed1-phase1 --loss_type ns --model mnist_dcgan --gpu 0 --seed 3 --major_ratio 0.9 --num_data 60000 --resample_score ldr_conf_5.0_ratio_50 --num_steps 20000 --p1_step 15000 --use_eval_logits 1
```

## Eval - MNIST-FMNIST
```
python eval_ae_score.py -d mnist_fmnist -r ./dataset/mnist_fmnist --baseline_exp_path exp_results/mf0.9-n60000-mnist_dcgan-bs64-loss_ns-seed1-inclusive --resample_exp_path exp_results/mf0.9-n60000-mnist_dcgan-bs64-loss_ns-seed1-inclusive --resample_score ldr_conf_3.0_ratio_50_beta_1.0 --use_loss --major_ratio 0.9 --num_data 60000 --seed 1 --name inclusive
```


## StyleGAN2
We use the implementation of https://github.com/rosinality/stylegan2-pytorch .
All the commands should be executed inside the stylegan2 directory.

1. Prepare data
Downlaod FFHQ dataset from https://github.com/NVlabs/ffhq-dataset
Then, convert to LMDB format. 
```
python prepare_data.py --out ./dataset/ffhq/lmdb_256.mdb --size 256 --path ./dataset/ffhq
```

2. Train - Phase 1
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=15694 train_ffhq.py --root ./dataset/ffhq/lmdb_256.mdb --batch 4 --dataset ffhq --exp_name ffhq-seed1 --seed 1
```

3. Train - Phase 2
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=15694 train_ffhq_phase2.py --root ./dataset/ffhq/lmdb_256.mdb --batch 4 --dataset ffhq  --exp_name ffhq-phase2-seed1 --baseline_exp_name ffhq-seed1 --seed 1 --resample_score ldr_conf_3.0_ratio_50
```

4. Evaluate
```
python eval_gan_drs.py -d ffhq -r ./dataset/ffhq/lmdb_256.mdb --exp_name stylegan2-ffhq-phase2 --model stylegan2 --seed 1 --netG_ckpt_step 250000 --gpu 0
```
