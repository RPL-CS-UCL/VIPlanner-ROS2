# Preperations

docker build -f docker/Dockerfile.eval --progress=plain -t viplanner-eval .

docker run --rm -it --gpus all -v "$(pwd)":/workspace -e PYTHONPATH=/workspace viplanner-eval bash

