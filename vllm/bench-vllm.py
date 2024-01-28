import sys, time
import pandas as pd
from vllm import SamplingParams, LLM

#from https://modal.com/docs/guide/ex/vllm_inference
import pandas as pd
import GPUtil
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = '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'
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


MODEL_DIR = "/workspace/vllm-models"

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    import os

    snapshot_download(
        "mistralai/Mistral-7B-Instruct-v0.2",
        local_dir=MODEL_DIR,
        token=os.environ["HUGGING_FACE_HUB_TOKEN"],
    )
    return LLM(MODEL_DIR)


def generate(question, llm, note=None):
    response = {'question': question, 'note': note}
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1,
        max_tokens=200,
    )
    
    start = time.perf_counter()
    result = llm.generate(question, sampling_params)
    request_time = time.perf_counter() - start
    gpu_nm = get_gpu_name()
    for output in result:
        response['time'] = request_time
        response['answer'] =  output.outputs[0].text
        response['compute'] =  gpu_nm
        response['word_cnt'] =  len(output.outputs[0].text)
        response['model'] =  'mistralai/Mistral-7B-Instruct-v0.2'
        response['inference_engine'] = 'vllm'
    
    return response

if __name__ == '__main__':
    llm = download_model_to_folder()
    counter = 1
    responses = []

    for q in questions:
        response = generate(question=q, llm=llm, note='vLLM')
        if counter >= 2:
            responses.append(response)
        counter += 1
    
    df = pd.DataFrame(responses)
    df.to_csv('bench-vllm.csv', index=False)