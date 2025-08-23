# If using Azure with mounted storage, do not set HF_HOME to a mounted path to avoid disk space errors
export HF_HOME=/.cache

# python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-8B')"

# # Move downloaded model files to the mounted path on verl_custom/hub/
# mv /.cache/hub/models--Qwen--Qwen3-8B verl_custom/hub/

# python3 -c "print('Qwen3-8B downloaded and moved to verl_custom/hub/ successfully')"


# python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-3B-Instruct')"

# # Move downloaded model files to the mounted path on verl_custom/hub/
# mv /.cache/hub/models--Qwen--Qwen2.5-3B-Instruct verl_custom/hub/

# python3 -c "print('Qwen2.5-3B-Instruct downloaded and moved to verl_custom/hub/ successfully')"


# python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-4B')"

# # Move downloaded model files to the mounted path on verl_custom/hub/
# mv /.cache/hub/models--Qwen--Qwen3-4B verl_custom/hub/

# python3 -c "print('Qwen3-4B downloaded and moved to verl_custom/hub/ successfully')"


python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-1.7B')"

# Move downloaded model files to the mounted path on verl_custom/hub/
mv /.cache/hub/models--Qwen--Qwen3-1.7B verl_custom/hub/

python3 -c "print('Qwen3-1.7B downloaded and moved to verl_custom/hub/ successfully')"


# python3 -c "import transformers; transformers.pipeline('text-generation', model='meta-llama/Llama-3.1-8B-Instruct')"

# # Move downloaded model files to the mounted path on verl_custom/hub/
# mv /.cache/hub/models--meta-llama--Llama-3.1-8B-Instruct verl_custom/hub/

# python3 -c "print('Llama-3.1-8B-Instruct downloaded and moved to verl_custom/hub/ successfully')"