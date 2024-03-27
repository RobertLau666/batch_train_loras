#!/bin/bash

# Set variables
root_path="/mnt/chenyu.liu/lora-scripts/train"
class_name="Genshin" # "OC"
need_train_person_lists=(
  # OC
  # [0]="Deredere Female"
  # [1]="Tsundere Male"
  
  # Genshin
  # [0]="Nahida girl"
  # [1]="Hutao girl"
  # [2]="RaidenShogun girl"
  # [3]="Yoimiya girl"
  # [4]="Tartaglia boy"
  # [5]="KamisatoAyaka girl"
  # [6]="YaeMiko girl"
  # [7]="Xiao boy"
  # [8]="Ganyu girl"
  # [9]="KaedeharaKazuha boy"
  # [10]="Zhongli boy"
  [0]="Albedo boy"
  [1]="Aloy girl"
  [2]="Amber girl"
  [3]="Barbara girl"
  [4]="Bennett boy"
  [5]="Beidou girl"
  [6]="Diona girl"
  [7]="Diluc boy"
  [8]="Dori girl"
  [9]="Faruzan girl"
  [10]="Fischl girl"
  [11]="AratakiItto boy"
  [12]="KujoSara girl"
  [13]="KukiShinobu girl"
  [14]="Kaveh boy"
  [15]="Kaeya boy"
  [16]="Candace girl"
  [17]="Collei girl"
  [18]="Klee girl"
  [19]="Keqing girl"
  [20]="Layla girl"
  [21]="Razor boy"
  [22]="Lisa girl"
  [23]="Wanderer boy"
  [24]="ShikanoinHeizou boy"
  [25]="Rosaria girl"
  [26]="Mika boy"
  [27]="Mona girl"
  [28]="Nilou girl"
  [29]="Ningguang girl"
  [30]="Noelle girl"
  [31]="Qiqi girl"
  [32]="Jean girl"
  [33]="Cyno boy"
  [34]="Sucrose girl"
  [35]="SangonomiyaKokomi girl"
  [36]="Shenhe girl"
  [37]="KamizatoAyato boy"
  [38]="Tighnari boy"
  [39]="Thoma boy"
  [40]="Venti boy"
  [41]="Gorou boy"
  [42]="Xiangling girl"
  [43]="Xinyan girl"
  [44]="Xingqiu boy"
  [45]="Yanfei girl"
  [46]="Yaoyao girl"
  [47]="Yelan girl"
  [48]="Lumine girl"
  [49]="Eula girl"
  [50]="YunJin girl"
  [51]="Sayu girl"
  [52]="Chongyun boy"
  
  # Test
)
start_id=0
train_cuda_id=3
test_cuda_id=3

for (( i="$start_id"; i<${#need_train_person_lists[@]}; i++ )); do
    # 分隔符IFS用于指定单词间的分隔符，这里使用空格
    IFS=' ' read -ra row <<< "${need_train_person_lists[i]}"
    person_name="${row[0]}"
    gender="${row[1]}"
    
    echo -e "\033[32m${i}/${#need_train_person_lists[@]}: $class_name $person_name $gender 正在处理...\033[0m"
    
    echo "新建white、out文件夹..."
    cd /mnt/chenyu.liu/Datasets/lora_data/"${class_name}"/"${person_name}"/
    if [ ! -d white ]
    then
        mkdir -p white
    fi
    if [ ! -d out ]
    then
        mkdir -p out
    fi
    
    echo "抠图，白背景..."
    cd /mnt/chenyu.liu/anime-segmentation
    python inference.py --net isnet_is --ckpt isnetis.ckpt --data /mnt/chenyu.liu/Datasets/lora_data/"${class_name}"/"${person_name}"/orig/ --out /mnt/chenyu.liu/Datasets/lora_data/"${class_name}"/"${person_name}"/white/ --only-matted
    
    echo "生成txt..."
    folder_path="/mnt/chenyu.liu/Datasets/lora_data/"${class_name}"/"${person_name}"/white/"
    for file in $(ls $folder_path); do
      # echo $file
      file_basename=$(basename $file)
      file_name="${file_basename%.*}"
      # echo $file_name
      txt_path="/mnt/chenyu.liu/Datasets/lora_data/"${class_name}"/"${person_name}"/out/$file_name.txt"
      prompt="1$gender, solo, $person_name, white_background"
      touch $txt_path
      echo $prompt > $txt_path
    done
    
    echo "复制数据..."
    dirs="${root_path}/${class_name}/${person_name}/6_${person_name}"
    if [ ! -d "$dirs" ]
    then
        mkdir -p "$dirs"
    fi
    cp -r /mnt/chenyu.liu/Datasets/lora_data/"$class_name"/"$person_name"/white/* "$dirs"
    cp -r /mnt/chenyu.liu/Datasets/lora_data/"$class_name"/"$person_name"/out/* "$dirs"
    
    echo "训练$person_name..."
    person_path="${root_path}/${class_name}/${person_name}"
    # Activate train venv
    cd /mnt/chenyu.liu/lora-scripts
    source venv/bin/activate
    # export CONDA_DEFAULT_ENV
    # echo $CONDA_DEFAULT_ENV

    # Train by train_batch_.sh
    export TRAIN_DATA_DIR="$person_path"
    export OUTPUT_NAME="$person_name"
    export CUDA_VERSION=120
    export CUDA_VISIBLE_DEVICES=$train_cuda_id
    bash train_batch_.sh
    deactivate
    
    echo "移动lora模型到webui目录下..."
    # Copy the trained lora model to lora directory on the webui, for later webui test
    cp -r /mnt/chenyu.liu/lora-scripts/output/"$person_name".safetensors /mnt/chenyu.liu/stable-diffusion-webui/models/Lora/

    echo "测试$person_name..."
    # Activate test env
    conda activate py39
    cd /mnt/chenyu.liu/LoRA_/comic_character_sd_service-main
    export PYTHONPATH=$(pwd):$PYTHONPATH
    cd test_scripts

    # Test by pipe_test_batch_.py
    export LORA_PATH="/mnt/chenyu.liu/lora-scripts/output/$person_name.safetensors"
    export GENDER="$gender"
    export CUDA_VISIBLE_DEVICES=$test_cuda_id
    python pipe_test_batch_.py
    
    echo ""
done
