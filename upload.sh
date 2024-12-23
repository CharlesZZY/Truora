#!/bin/bash

# Define the directory where the files are located and the file type pattern
directory="./datasets/CBU0521DD_stories/"  # 目标目录，包含要上传的文件
file_extension="*.wav"  # 目标文件类型，这里是所有的 .wav 文件

# Change to the target directory (进入目标目录)
cd "$directory" || { echo "Directory $directory does not exist"; exit 1; }  # 如果目录不存在，输出错误并退出

# Get the list of files (获取符合条件的所有文件)
files=($file_extension)  # 获取目录下所有符合 *.wav 文件扩展名的文件列表

# Set the batch size (设置每次提交的文件数量)
batch_size=4  # 每次提交的文件数量为 4
total_files=${#files[@]}  # 获取总文件数

# Loop through the files and perform commit and push in batches
# 遍历文件列表，每批次处理 batch_size 个文件
for ((i=0; i<$total_files; i+=batch_size)); do
  # Get the current batch of files (获取当前批次的文件)
  batch_files=("${files[@]:$i:$batch_size}")  # 从文件列表中截取出当前批次的文件

  # Print the current batch of files (打印当前批次的文件)
  echo "Batch $((i / batch_size + 1)) - Files to be uploaded: ${batch_files[*]}"  # 打印当前批次的文件名

  # Stage the files for commit (将当前批次的文件添加到 Git 暂存区)
  git add "${batch_files[@]}"  # 将当前批次的文件添加到 Git 索引中

  # Create the commit message (生成提交信息)
  commit_message="Batch $((i / batch_size + 1)) - Upload files: ${batch_files[*]}"  # 使用当前批次的文件列表生成提交信息

  # Commit the changes with the generated message (提交更改，使用批次信息作为提交消息)
  git commit -m "$commit_message"  # 执行提交，提交信息包含批次信息和文件列表

  # Push the changes to the remote repository (将更改推送到远程仓库)
  git push origin master  # 将本地更改推送到远程 Git 仓库，推送到 'master' 分支（根据需要修改分支名称）

done  # 循环结束