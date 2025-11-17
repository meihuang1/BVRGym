from UtilsLib.fleet import Fleet
import matplotlib.pyplot as plt
from jsb_gym.TAU.config import f16_traj_track as cfg

def main():
    # realtime=True: 开启实时3D可视化（每架飞机独立窗口）
    # realtime=False: 只记录数据，仿真结束后统一显示
    # record_debug=True: 记录控制数据用于生成误差图
    fleet = Fleet(conf=cfg, realtime=False, record_debug=True)

    k_max = 60_000
    print(f"开始仿真，共 {k_max} 步...")
    for k in range(k_max):
        if k % 1000 == 0:
            print(f"进度: {k}/{k_max} ({k/k_max*100:.1f}%)")
        fleet.step(k)

    print("\n" + "="*60)
    print("仿真完成！正在生成可视化...")
    print("="*60)
    
    # ========== 方式1：静态航迹对比图（最快，推荐） ========== #
    print("\n[1/3] 生成静态航迹对比图...")
    fleet.plot_all_static()
    
    # ========== 方式2：双窗口动画回放（轨迹+姿态） ========== #
    print("\n[2/3] 生成双窗口动画回放...")
    fleet.animate(interval=20, ac_idx=0, save_gif=True, 
                 gif_filename1='trajectory_animation.gif', 
                 gif_filename2='attitude_animation.gif')  
    # 参数说明:
    #   interval: 帧间隔（毫秒），越小播放越快（10-50比较合适）
    #   ac_idx: 显示哪架飞机的姿态（0=第1架，1=第2架...）
    #   save_gif: True保存为GIF文件，False只显示不保存
    #   gif_filename1: 轨迹动画GIF文件名
    #   gif_filename2: 姿态动画GIF文件名
    
    # ========== 方式3：详细控制调试图（可选） ========== #
    # 显示每架飞机的位置/速度误差、控制指令等
    # 如果飞机数量多，会生成很多图，按需开启
    print("\n[3/3] 生成控制调试图...")
    # fleet.show_all_final()
    
    print("\n" + "="*60)
    print("所有可视化完成！")
    print("="*60)

if __name__ == '__main__':
    main()
