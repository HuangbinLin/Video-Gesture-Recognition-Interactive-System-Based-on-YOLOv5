import sys
import pygame
import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from detect import yolov5
import cv2
import copy

class MusicPlayer:
    def __init__(self,video_source=0):
        self.folder_path = "./music"
        self.current_song = None
        self.window = None
        self.open = 0
        self.times = 0


        self.class_names = ['invalid', 'up', 'down', 'left', 'right', 'close','small','big']
        self.pose_list = ['invalid']
        self.prev_frame = 'invalid'
        self.cur_frame = 'invalid'
        self.fangzhi = 0
        self.shangci = 0
        self.result = 0
        self.duration = 0 #持续时间

        self.photo = None
        self.video_source = video_source
        self.video_capture = cv2.VideoCapture(self.video_source)

        # 加载模型
        self.capture = 'test.mp4'
        self.weights = "best.onnx"
        root = os.path.dirname(os.path.abspath(__file__))
        self.model_yolov5_path = os.path.join(root, self.weights)
        self.device = 'cpu'  # 大赛后台使用CPU判分
        self.model = yolov5(self.model_yolov5_path, confThreshold=0.8, nmsThreshold=0.6, objThreshold=0.5)
        self.color = (0, 255, 0)  # 框的颜色（BGR格式）

        # 创建界面
        # self.model = 
        self.create_player()

        self.update()
        self.start()
    

    # 播放音乐
    def play_music(self, file_name):
        file_name = file_name + '.mp3'
        file_path = os.path.join(self.folder_path, file_name)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

    def popup_destroy(self):
        if(self.open ==1):
            self.popup.destroy()
            self.open = 0
    

    def popup_window(self):
        if(self.open == 0):
            self.popup = tk.Toplevel(self.window)
            self.popup.title("歌词")


            # 计算弹出窗口的位置
            self.window.update_idletasks()  # 确保主窗口的几何信息已更新
            # x = self.window.winfo_x() + (self.window.winfo_width() // 2) - (self.popup.winfo_width() // 4)  # 主窗口中心的横坐标减去弹出窗口宽度的一半
            x = self.window.winfo_x() + (self.window.winfo_width() // 8) - (self.popup.winfo_width() // 4)   # 主窗口中心的横坐标减去弹出窗口宽度的一半
            y = self.window.winfo_y() + (self.window.winfo_height() // 2) - (self.popup.winfo_height() // 2)  # 主窗口中心的纵坐标减去弹出窗口高度的一半
            self.popup.geometry(f"400x200+{x}+{y}")  # 设置弹出窗口的大小和位置
        
            # 创建文本标签
            self.label = tk.Label(self.popup, text="歌词")
            self.label.pack(padx=10, pady=10)
            self.open = 1
        

    # 暂停
    def stop_music(self):
        pygame.mixer.music.stop()

    # 切换到上一首，内嵌入play_music
    def play_previous_song(self, playlist, current_index):
        # 如果是最后一首，则返回到第一首
        if current_index > 0:
            current_index -= 1
        else:
            current_index = playlist.size() - 1

        selected_song = playlist.get(current_index)
        self.play_music(selected_song)
        playlist.select_clear(0, tk.END)
        playlist.select_set(current_index)

    # 切换到下一首，内嵌入play_music
    def play_next_song(self, playlist, current_index):
        # 是否到达最后一首，是的话回到第一首
        if current_index < playlist.size() - 1:
            current_index += 1
        else:
            current_index = 0

        selected_song = playlist.get(current_index)
        self.play_music(selected_song)
        # 清除上一首的可视化表达
        playlist.select_clear(0, tk.END)
        # 可视化列表
        playlist.select_set(current_index)

    # 调整音量
    def set_volume(self, volume):
        volume = float(volume)
        pygame.mixer.music.set_volume(volume / 100)  # 将滑块值转换为音量值（0.0-1.0）


    def update(self):
        ret, frame = self.video_capture.read()
        if ret:
            org_fram = copy.copy(frame)
            boxes,classIds,confidences = self.model.detect(org_fram)
            if len(boxes) != 0:
                for box, classId, confidence in zip(boxes, classIds, confidences):
                    x,y,w,h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, thickness=2)
                    pred_name = self.class_names[classId]  # 可能有无效手势
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (255, 0, 0)  # 文本颜色，BGR格式
                    thickness = 2  # 文本线条宽度
                    cv2.putText(frame, pred_name, (x,y), font, font_scale, font_color, thickness)

                    self.cur_frame = pred_name
                    if (self.cur_frame != self.prev_frame):
                        self.duration = 1
                    else:
                        self.duration += 1
                    self.prev_frame = self.cur_frame

                    # if (self.shangci == self.cur_frame and self.fangzhi <3) :
                    #     self.fangzhi += 1
                    #     self.cur_frame = "invalid"
                    # else:
                    #     self.fangzhi = 0
                    # print(self.fangzhi)


                    if(self.cur_frame == 'invalid' and self.duration > 3):
                        self.pose_list.append(self.cur_frame)
                    elif(self.cur_frame != 'invalid' and self.duration > 1):
                        self.pose_list.append(self.cur_frame)

                    
                    if(len(self.pose_list) >= 2):
                        # 如果是上滑
                        if(self.pose_list[-2] == 'down' and self.pose_list[-1] == 'up'):
                            self.result = 1
                        # 如果是下滑
                        elif(self.pose_list[-2] == 'up' and self.pose_list[-1] == 'down'):
                            self.result = 2
                        # 如果是左滑 
                        elif(self.pose_list[-2] == 'right' and self.pose_list[-1] == 'left'):
                            self.result = 3
                        # 如果是右滑
                        elif(self.pose_list[-2] == 'left' and self.pose_list[-1] == 'right'):
                            self.result = 4
                        # 如果是打开
                        elif(self.pose_list[-2] == 'close' and self.pose_list[-1] == 'up'):
                            self.result = 5
                        # 如果是关闭
                        elif(self.pose_list[-2] == 'up' and self.pose_list[-1] == 'close'):
                            self.result = 6
                        # 放大
                        elif(self.pose_list[-2] == 'small' and self.pose_list[-1] == 'big'):
                            self.result = 7
                        # small
                        elif(self.pose_list[-2] == 'big' and self.pose_list[-1] == 'small'):
                            self.result = 8



                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # 上下左右打关
        # 上滑
        if(self.result == 1):
            self.shangci = "up"
            current_volume = self.volume_scale.get()
            if(current_volume + 10 >100):
                current_volume = 100
            else:
                current_volume = current_volume + 10
            self.volume_scale.set(current_volume)
            pygame.mixer.music.set_volume(current_volume/100)

            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        # 下滑
        elif(self.result == 2):
            self.shangci = "down"
            current_volume = self.volume_scale.get()
            if(current_volume - 10 < 0):
                current_volume = 0
            else:
                current_volume = current_volume - 10
            self.volume_scale.set(current_volume)
            pygame.mixer.music.set_volume(current_volume/100)
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        # 左滑 上一首
        elif(self.result == 3):
            self.shangci = "left"
            self.play_previous_song(self.playlist, self.playlist.curselection()[0])
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        # 右滑 下一首
        elif(self.result == 4):
            self.shangci = "right"
            self.play_next_song(self.playlist, self.playlist.curselection()[0])
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        # 打开 播放
        elif(self.result == 5):
            self.shangci = "open"
            self.play_music(self.playlist.get(self.playlist.curselection()[0]))
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        # 关闭 暂停
        elif(self.result == 6):
            self.shangci = "close"
            self.stop_music()
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        elif(self.result == 7):
            self.shangci = "big"
            self.popup_window()
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0
        elif(self.result == 8):
            self.shangci = "small"
            self.popup_destroy()
            self.result = 0
            self.pose_list = ['invalid']
            self.window.after(10, self.update_only)
            return 0


        self.window.after(15, self.update)

    def update_only(self):
        ret, frame = self.video_capture.read()
        self.times += 1
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        
        if(self.times ==  50):
            self.window.after(15, self.update)
            self.times = 0
            return 0
        self.window.after(10, self.update_only)

    # 创建窗口和各类组件
    def create_player(self):
        # 初始化声音设备
        pygame.mixer.init()
        self.current_song = None

        # 创建主窗口
        self.window = tk.Tk()
        self.window.title("音乐播放器")

        image = Image.open("pic/music.png")
        image = image.resize((120, 120))  # 调整图片大小
        self.photo_music = ImageTk.PhotoImage(image)
        # 创建Label，并显示图片
        self.image_label = tk.Label(self.window, image=self.photo_music, width=200, height=200)
        self.image_label.grid(row=0, column=0, rowspan=2, columnspan=2)

        # 创建播放列表
        self.playlist = tk.Listbox(self.window, selectbackground="#4286f4", selectforeground="white", width=28)
        self.playlist.grid(row=0, column=2, rowspan=2, columnspan=2)
        # music文件
        mp3_files = [file for file in os.listdir(self.folder_path) if file.endswith(".mp3")]
        for file in mp3_files:
            file = file.split('.')[0]
            # 遍历嵌入
            self.playlist.insert(tk.END, file)

        # 设置默认选择第一首歌曲
        self.playlist.select_set(0)
        # 根据列表点击的位置，定义目前的歌曲
        current_song = self.playlist.get(self.playlist.curselection()[0])

        #创建视频显示窗口
        self.video_canvas = tk.Canvas(self.window, width=400, height=300)
        # self.video_canvas.grid(row=0, column=8, rowspan=5, columnspan=3)
        self.video_canvas.grid(row=0, column=8, rowspan=5)


        # 创建下一首按钮
        image = Image.open("pic/Playernext.png")
        image = image.resize((90, 90))  # 调整图片大小
        self.photo_Playernext = ImageTk.PhotoImage(image)

        self.next_button = tk.Button(self.window, image=self.photo_Playernext, width=100, height=100, command=lambda: self.play_next_song(self.playlist, self.playlist.curselection()[0]))
        self.next_button.grid(row=3, column=3)


        # 创建上一首按钮
        image = image.rotate(180)
        self.photo_Playerprev = ImageTk.PhotoImage(image)

        self.previous_button = tk.Button(self.window, image=self.photo_Playerprev, width=100, height=100, command=lambda: self.play_previous_song(self.playlist, self.playlist.curselection()[0]))
        self.previous_button.grid(row=3, column=0)


        # 创建播放按钮
        image = Image.open("pic/Playerplay.png")
        image = image.resize((90, 90))  # 调整图片大小
        self.photo_Playerplay = ImageTk.PhotoImage(image)

        self.play_button = tk.Button(self.window, image=self.photo_Playerplay, width=100, height=100, command=lambda: self.play_music(current_song))
        self.play_button.grid(row=3, column=1)


        # 创建停止按钮
        image = Image.open("pic/Playerstop.png")
        image = image.resize((90, 90))  # 调整图片大小
        self.photo_Playerstop = ImageTk.PhotoImage(image)
        
        self.stop_button = tk.Button(self.window, image=self.photo_Playerstop, width=100, height=100, command=self.stop_music)
        self.stop_button.grid(row=3, column=2)

        # 创建音量滑块，滑块值自动传入到set_volume
        self.volume_scale = tk.Scale(self.window, from_=100, to=0, orient="vertical", command=self.set_volume, length=300)
        self.volume_scale.grid(row=0, column=4, rowspan=4)
        self.volume_scale.set(50)
        pygame.mixer.music.set_volume(0.5)

        # 创建弹出窗口
        # self.popup = tk.Toplevel()
        # self,popup.title("弹出框")

        # 设置窗口大小和位置
        # 470是横向的
        self.window.geometry("900x310")
        self.window.resizable(False, False)
        self.window.configure(bg="#f9f9f9")
        self.window.eval('tk::PlaceWindow %s center' % self.window.winfo_toplevel())


    def start(self):
        if self.window:
            self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.window.mainloop()

    def on_closing(self):
        self.video_capture.release()  # 释放视频捕捉资源
        self.window.destroy()  # 关闭Tkinter窗口
        sys.exit()  # 退出程序