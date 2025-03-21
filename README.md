# Train a GPT2 model _for free_ on Google TPUs using JAX

This repository contains JAX code for pretraining a GPT2 model on Google TPUs. The notebook is adapted from the offical [miniGPT tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html) and works on free-tier Colab, Kaggle and Cloud TPU (single board TPU v2 - v6e all work).

Only GPT2 and GPT2 medium have been tested. Bigger variants will OOM on TPU v3. Dataset used is [OpenWebText](https://www.kaggle.com/datasets/windmaple/openwebtext-gpt2), same as [nanoGPT](https://github.com/karpathy/nanoGPT/) because I wanted to compare the final losses.

<img width="600" alt="Screenshot 2025-02-25 at 09 59 00" src="GPT2.png" />


## How to run
First, get your Kaggle credential and Weights and Biases API key ready. Add to your secrets if you are using Colab and Kaggle.

### Colab
1. Manully download the OpenWebText dataset using Kaggle CLI (you need to set up [kaggle.json](https://www.kaggle.com/docs/api) first) like this:
   `kaggle datasets download -d windmaple/OpenWebText-gpt2 && unzip OpenWebText-gpt2.zip`
2. I store the `.bin` files on my Google Drive (`/content/drive/MyDrive/LLM-pretraining/OpenWebText/`) so that they are cached. Change `data_dir` if your `.bin` files are in a different folder.
3. Make sure your W&B API key is accessible to your notebook
4. Connect the TPU v2 runtime and run. Free tiers gives you an hour or two before disconnecting, so you can't really finish the training
5. A paid account, you may be able to see it through, although I haven't tried it myself (TPU v2 is just too slow). Colab now offers TPU v5e as well, but it's only one chip (unlike v2's 4 chip) and you need to change the mesh to run on it

### Kaggle
Kaggle is more generous, offering 9 hours of non-interrupted TPU v3 per session, which is sufficient to train the smallest GPT2 variant.
1. Import the notebook on Kaggle
2. Add the [OpenWebText](https://www.kaggle.com/datasets/windmaple/openwebtext-gpt2) dataset as input in the top right corner of the side panel on the right.
3. Make sure W&B API key is accessible to your notebook
4. Choose TPU v3
5. Run the notebook. Kaggle will first download the dataset first. After that, it takes ~7 hours to finish. 
6. You can also try GPT2 medium if you change the `GPT2_variant` variable. Kaggle will stop the run before it finishes though

An alternative way to run is to `Save version` -> `Save & Run ALL`, which just run the notebook in the background.

Technically you can also train a GPT2 medium model on Kaggle; although Kaggle disconnects you after 9 full hours it can save checkpoint files for you, so that you can resume training. But I haven't tried this because it's a bit of a pain.

### Cloud TPU
OK, I lied. This one is not free. But since you are the paying God, you can pretty much do whatever you want, like training GPT2 medium to completion. 
1. Spin up your TPU VM and ssh into it
2. Download the notebook
3. Pip install kaggle and get your [kaggle.json](https://www.kaggle.com/docs/api)
4. Pip install jupyter
5. Start a tmux or screen session and then run the notebook like this (an alternative is to convert it to a `.py` file):
   `export WANDB_API_KEY=$your_key; time jupyter execute GPT2_pretrain.ipynb`
6. There won't be much logging shown in the console, but don't worry, everything is directed to your W&B so you can see the output there

## Monitoring TPU usage

### Cloud TPU
W&B has integration with Cloud TPU and reports [TPU metrics](https://docs.wandb.ai/guides/models/app/settings-page/system-metrics/#google-cloud-tpu) in the systems panel (2nd page) automatically. 

![Screenshot 2025-02-24 at 16 35 04](https://github.com/user-attachments/assets/25abeacc-cea1-418d-b55c-039ca2eb7851)

You can also, in another console, `pip install tpu-info` and then `watch -n 1 tpu-info`.
                                                                                                                     

![Screenshot 2025-02-24 at 16 33 49](https://github.com/user-attachments/assets/015ac095-dcee-48ac-8c97-60a62b6e1e3a)


Google Cloud console has additional monitoring tools if you use v4 or newer.

### Colab and Kaggle
Neither is integrated with W&B unfortunately. But you can still `pip install tpu-info` and then add `!tpu-info` in the middle of the training loop. Note that this might slow down training a bit.


## Final result
If stars are aligned, you can get the final losses like below:

<img width="359" alt="Screenshot 2025-02-24 at 16 37 28" src="https://github.com/user-attachments/assets/fb08ecd2-cfb5-42a4-bf62-187bf0cec764" />

which are very much in line with [nanoGPT's](https://github.com/karpathy/nanoGPT/).

## Faster speed on Trillium
Trillium chips, which have 32G HBM and accommodate 2X batch size, can finish training in 82 minutes. 
