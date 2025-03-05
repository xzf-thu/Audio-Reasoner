
# Audio-Reasoner
<p align="center">
    <img src="assets\title.png" width="80%"/>
</p>

## Abstract
We implemented inference scaling on **Audio-Reasoner**, a large audio language model, enabling **deepthink** and **structured chain-of-thought (COT) reasoning** for multimodal understanding and reasoning. To achieve this, we constructed CoTA, a high-quality dataset with **1.2M reasoning-rich samples** using structured COT techniques. Audio-Reasoner achieves state-of-the-art results on **MMAU-mini(+25.42%)** and **AIR-Bench-Chat(+14.57%)** benchmarks.

<p align="center">
Audio-Reasoner-7B <a href="https://huggingface.co"></a>🤗 (coming soon) |  CoTA Dataset <a href="https://huggingface.co"></a> 🤗 (coming soon)<br>
Paper <a href="https://arxiv.org/abs/2503.02318"> 📑</a> ｜ Wechat <a href="https://github.com/xzf-thu/Audio-Reasoner/blob/main/assets/wechat.jpg">💭</a> ｜ Code <a href="https://github.com/xzf-thu/Audio-Reasoner"> ⚙️</a>
<br>
<a href="#demo"> Demo</a> • <a href="#install">Install</a> • <a href="#quick-start">Quick Start</a> • <a href="#faq">FAQ</a> • <a href="#contact">Contact us</a><br>
<br>
If you like us, pls give us a star⭐ !
</p>



## Main Results
<p align="center">
    <img src="assets\main_result.png" width="100%"/>
</p>




## News and Updates
- **2025.03.05:** ✅**Audio-Reasoner Paper is uploaded to arXiv.**
- **2025.03.04:** ✅**Demos, inference code and evaluation results have been released.**
- **2025.03.04:** ✅**Create this repo.**

## Roadmap
- **2025.03:** **🔜Release Audio-Reasoner-7B checkpoint as well as the evaluation code.**

- **2025.03:** **🔜Upload CoTA dataset to HuggingFace🤗.**

- **2025.04:** **🔜Open-source data systhesis pipeline and training code**.

## Demo
<p align="center" width="80%">
<video controls src="https://github.com/user-attachments/assets/d50f75e7-288b-454b-92a3-c6f058be231b" title="v" width="100%"></video>
</p>

## Features
✅ Audio-Reasoner enables **deep reasoning and inference scaling** in audio-based tasks, built on Qwen2-Audio-Instruct with structured CoT training.

✅ CoTA offers **1.2M** high-quality captions and QA pairs across domains for structured reasoning and enhanced pretraining. 

✅ Pretrained model and dataset encompassing various types of audio including sound, music, and speech, has achieved state-of-the-art results across multiple benchmarks. Refer to our <a href="https://arxiv.org/abs/2503.02318">paper</a> for details.


## Install

**Clone and install**

- Clone the repo
``` sh
git clone https://github.com/xzf-thu/Audio-Reasoner.git

cd Audio-Reasoner
```

- Install the required packages
```sh
conda create -n Audio-Reasoner python=3.10
conda activate Audio-Reasoner

pip install -r requirements.txt
pip install transformers==4.49.1
```

## Quick Start

**Chat using ms-swift**
```sh
import os
import re
from typing import List, Literal
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset, get_template
from swift.plugin import InferStats


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=2048, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    output = ""
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
        output += resp_list[0].choices[0].delta.content
    print()
    print(f'metric: {metric.compute()}')
    return output


def get_message(audiopath, prompt):
    messages = [
        {"role": "system", "content": system},
        {
        'role':
        'user',
        'content': [{
            'type': 'audio',
            'audio': audiopath
        }, {
            'type': 'text',
            'text':  prompt
        }]
    }]
    return messages

system = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'
infer_backend = 'pt'
model = 'qwen2_audio'
last_model_checkpoint = "" #Please replace it with the path to checkpoint
engine = PtEngine(last_model_checkpoint, max_batch_size=64,  model_type = model)

def audioreasoner_gen(audiopath, prompt):
    return infer_stream(engine, InferRequest(messages=get_message(audiopath, prompt)))

def main():
    #Please replace it with your test aduio 
    audiopath = "assets/test.wav" 
    #Please replace it with your questions about the test aduio    
    prompt = "Which of the following best describes the rhythmic feel and time signature of the song?"  
    audioreasoner_gen(audiopath, prompt)

if __name__ == '__main__':
    main()
```

**Local test**

```sh
conda activate Audio-Reasoner
cd Audio-Reasoner
# test run the preset audio samples and questions
python inference.py 
```

## FAQ

**1. What kind of audio can Audio - Reasoner understand and what kind of thinking does it perform?**
Audio - Reasoner can understand various types of audio, including sound, music, and speech. It conducts in - depth thinking in four parts: **planning, caption, reasoning, and summary**.

**2. Why is transformers installed after 'ms-swift' in the environment configuration?**
The version of transformers has a significant impact on the performance of the model. We have tested that version `transformers==4.49.1` is one of the suitable versions. Installing ms-swift first may ensure a more stable environment for the subsequent installation of transformers to avoid potential version conflicts that could affect the model's performance.

## More Cases
<p align="center">
    <img src="assets\figure2-samples.png" width="90%"/>
</p>


##  Contact 

If you have any questions, please feel free to contact us via `zhifei001@e.ntu.edu.sg`.

##  Citation 
Please cite our paper if you find our model and detaset useful. Thanks! 
```
@misc{xie2025audioreasonerimprovingreasoningcapability,
      title={Audio-Reasoner: Improving Reasoning Capability in Large Audio Language Models}, 
      author={Zhifei Xie and Mingbao Lin and Zihang Liu and Pengcheng Wu and Shuicheng Yan and Chunyan Miao},
      year={2025},
      eprint={2503.02318},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2503.02318}, 
}
```



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=xzf-thu/Audio-Reasoner&type=Date)]([https://star-history.com/#xzf-thu/Audio-Reasoner&Date](https://star-history.com/#xzf-thu/Audio-Reasoner&Timeline))
