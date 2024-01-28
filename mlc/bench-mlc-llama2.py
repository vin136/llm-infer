from mlc_chat import ChatModule, ChatConfig
from mlc_chat.callback import StreamToStdout
from transformers import AutoTokenizer
import time
import sys
#sys.path.append('../../common/')
#from questions import questions
import pandas as pd
import GPUtil

def get_gpu_name():
    try:
        # Get the list of available GPUs
        gpus = GPUtil.getGPUs()

        if gpus:
            # Assuming you want the name of the first GPU
            gpu_name = gpus[0].name
            return gpu_name
        else:
            return "No GPU available."

    except Exception as e:
        return f"Error: {str(e)}"
        
questions = [
    # Literature
    "I want a hamburger",
    "Write a poem like Jordan Peterson.",
    "Write a poem like Tom Cruise from mission impossible.",
    "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
    "Who does Harry turn into a balloon?",
    "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
    # Math
    "What is the product of 9 and 8?",
    "If a train travels 120 kilometers in 2 hours, what is its average speed?",
    "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
]

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

cfg = ChatConfig(max_gen_len=200)
#cm = ChatModule(model="Llama-2-7b-chat-hf-q4f16_1", chat_config=cfg)


cm = ChatModule(
   model="dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
   model_lib_path="dist/prebuilt_libs/Llama-2-7b-chat-hf/Llama-2-7b-chat-hf-q4f16_1-cuda.so",
    chat_config=cfg
)

def tok_count(prompt:str):
    inputs = tokenizer(prompt)
    return len(inputs['input_ids'])

def predict(prompt:str):
    start_time = time.perf_counter()
    output = cm.generate(prompt=prompt)
    request_time = time.perf_counter() - start_time
    #'tok_count': tok_count(output)
    gpu_nm = get_gpu_name()
    return {'time': request_time,
            'question': prompt,
            'answer': output,
            'compute': gpu_nm,
            'word_cnt': len(output.split()),
            'model': 'meta-llama/Llama-2-7b-chat-hf',
            'inference_engine':'mlc',
            'note': 'mlc chat 7b q4f16_1'}

if __name__ == '__main__':
    counter = 1
    responses = []
    for q in questions:
        if counter >= 2: responses.append(predict(q))
        counter += 1

    df = pd.DataFrame(responses)
    df.to_csv('bench-mlc.csv', index=False)




