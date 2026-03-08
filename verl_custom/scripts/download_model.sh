# If using Azure with mounted storage, do not set HF_HOME to a mounted path to avoid disk space errors
export HF_HOME=/.cache

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen3-4B-Instruct-2507')"

# Move downloaded model files to the mounted path on hub/
mv /.cache/hub/models--Qwen--Qwen3-4B-Instruct-2507 verl_custom/hub/

python3 -c "print('Qwen3-4B downloaded and moved to hub/ successfully')"
