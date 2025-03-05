import os
from typing import List, Literal
import re
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
