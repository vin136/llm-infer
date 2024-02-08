SHELL = /bin/bash

#
.PHONY: allinstall
allinstall:	mlc
	source venv/bin/activate && \
	python3 -m pip install flash-attn --no-build-isolation && \
	python3 -m pip install -r requirements.txt




#mlc
.PHONY:	mlc
mlc: venv
	source venv/bin/activate && \
	python3 -m pip install git-lfs && \
	python3 -m pip install --pre --force-reinstall mlc-ai-nightly-cu118 mlc-chat-nightly-cu118 -f https://mlc.ai/wheels && \
	mkdir -p dist && \
	git clone https://github.com/mlc-ai/binary-mlc-llm-libs.git dist/prebuilt_libs && \
	git clone https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC dist/Llama-2-7b-chat-hf-q4f16_1-MLC && \
	git clone https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC dist/Mistral-7B-Instruct-v0.2-q4f16_1-MLC


#venv
.PHONY:	venv
venv:	
	cd /workspace && \
	python3 -m venv venv && \
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel 

# Cleaning
.PHONY: clean
clean: 	
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*
