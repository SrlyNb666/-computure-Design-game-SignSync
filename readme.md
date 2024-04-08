1、创建ananda的虚拟环境
    1\ conda creat -n sysb python==3.10.8
    2\ conda activate sysb
    3\ pip install -r requirements.txt
    4\ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
2、运行
    在sysb虚拟环境下运行demo.py文件即可，运行前需要在电脑上连接两个摄像头并分别放置在手腕的手背与手掌处。
    关于模拟键盘输入，需要鼠标点击文本框并且手做出相应的手语，当前我们的训练集只包括“A”到“G”,以及数字"0"到"10"。
3、停止运行
    按下键盘上的q即可停止运行