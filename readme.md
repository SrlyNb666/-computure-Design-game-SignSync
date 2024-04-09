1、创建anaconda的虚拟环境:
    1\ conda creat -n sysb python==3.10.8
    2\ conda activate sysb
    3\ pip install -r requirements.txt
    4\ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
2、运行
    一、如果直接使用终端cmd打开，则：
    1、新建文件夹 sysb，使用终端cd 到该文件夹，输入命令 git clone   https://github.com/SrlyNb666/computure-Design-game-SignSync.git 
    如果下载不了，则输入git clone https://gitee.com/srlynb/sysb.git
    如果使用第一个命令则输入cd computure-Design-game-SignSync
    如果使用第二个命令则使用 cd sysb
    2、 conda activate sysb
    3、python demo.py
    注：运行前需要在电脑上连接两个摄像头并分别放置在手腕的手背与手掌处。

    同时请查看demo.py的138到144行，按照参考画面示范里的图片视角要求，调整这两行代码，有选择性的注释或者取消注释，使得画面种手部视角与参考图一致。否则将非常影响模型精度


    关于模拟键盘输入，需要鼠标点击文本框并且手做出相应的手语，当前我们的训练集只包括“A”到“G”,以及数字"0"到"10"。
    二、使用vscode打开项目文件夹时安装python扩展，然后在右下角选择使用sysb环境，然后进入demo.py点击右上角 ”运行python程序“即可运行
    在sysb虚拟环境下运行demo.py文件即可，运行前需要在电脑上连接两个摄像头并分别放置在手腕的手背与手掌处。
    关于模拟键盘输入，需要鼠标点击文本框并且手做出相应的手语，当前我们的训练集只包括“A”到“G”,以及数字"0"到"10"。
    三、打开出现摄像头画面窗口后，按下空格开始识别手势并且输入文字到鼠标焦点下的文本框，再次按下空格就停止这一次的连续识别。
3、停止运行
    按下键盘上的q即可停止运行


