import torch
file_path = "outputs/2023-10-15/14-21-48-304097/checkpoints/val/loss.ckpt"

loss_state = torch.load(file_path)
print(loss_state)

# 打开一个文件以供写入
with open("loss.txt", "w") as file:
    # 使用 print 函数将内容输出到文件
    print(loss_state, file=file)
# 文件会在退出上下文管理器后自动关闭
