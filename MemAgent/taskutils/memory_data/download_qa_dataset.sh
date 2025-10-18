if [ -z $(which aria2c) ]; then
    sudo apt update
    yes | sudo apt install aria2
fi
echo "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json  
    out=squad.json
http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json  
    out=hotpotqa_dev.json" > __download.txt
aria2c -x 10 -s 10 -j 2 -i __download.txt
rm __download.txt