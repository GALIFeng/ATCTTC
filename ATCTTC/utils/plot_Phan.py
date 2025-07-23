import matplotlib.pyplot as plt
import numpy as np

def plot_slice(c):
    '''
    绘制3D图像c的切片图，x,y,z切换不同的显示轴，鼠标滚轮翻动切片
    '''

    # 计算提取区域的起始和结束索引

    # 归一化到 [0,1]
    C_min, C_max = c.min(), c.max()
    c = (c - C_min) / (C_max - C_min)

    # 使用可变列表来保存当前轴和索引
    slice_axis = [0]  # 0 -> z, 1 -> y, 2 -> x
    slice_idx  = [0]

    # 初始绘图
    fig, ax = plt.subplots()
    im = ax.imshow(
        np.real(c[slice_idx[0], :, :]),
        interpolation="none",
        vmin=0, vmax=1,
        cmap="gray"
    )
    ax.set_title(f"Axis: {slice_axis[0]}, Slice: {slice_idx[0]}")
    fig.colorbar(im, ax=ax)

    def update_slice():
        a = slice_axis[0]
        i = slice_idx[0]
        if a == 0:
            data = np.real(c[i, :, :])
        elif a == 1:
            data = np.real(c[:, i, :])
        else:  # a == 2
            data = np.real(c[:, :, i])
        im.set_data(data)
        ax.set_title(f"Axis: {a}, Slice: {i}")
        fig.canvas.draw_idle()

    def on_scroll(event):
        # 向上滚动增加索引，向下减少
        n = c.shape[slice_axis[0]]
        if event.button == 'up':
            slice_idx[0] = (slice_idx[0] + 1) % n
        elif event.button == 'down':
            slice_idx[0] = (slice_idx[0] - 1) % n
        update_slice()

    def on_key(event):
        key = event.key.lower()
        if key == 'z':
            slice_axis[0] = 2
        elif key == 'y':
            slice_axis[0] = 1
        elif key == 'x':
            slice_axis[0] = 0
        # 切换轴后重置索引
        slice_idx[0] = 0
        update_slice()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


    

