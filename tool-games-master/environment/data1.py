import pygame
import numpy as np
import random

# --- 配置 ---
WIDTH, HEIGHT = 600, 600
BG_COLOR = (255, 255, 255)
# 提取自你提供的代码中的颜色
COLOR_PINK = (255, 0, 255)   # 工具1
COLOR_CYAN = (0, 255, 255)   # 工具2
COLOR_YELLOW = (225, 225, 0) # 工具3
BLOCK_COLOR = (0, 0, 0)
GROUND_COLOR = (144, 238, 144) # 浅绿色
PLATFORM_COLOR = (0, 0, 255)
RED_BALL_COLOR = (255, 0, 0)

def create_soft_blob(color, radius, max_alpha=50):
    """
    创建一个带有径向渐变透明度的 Surface，用于模拟高斯分布的视觉效果。
    """
    # 创建一个支持 Alpha 通道的 Surface
    surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    
    # 从中心向外画同心圆，透明度逐渐降低
    steps = 20
    for i in range(steps):
        r = int(radius * (1 - i / steps))
        # 核心更不透明，边缘更透明
        alpha = int(max_alpha * (1 - i / steps))
        pygame.draw.circle(surf, (*color, alpha), (radius, radius), r)
    
    return surf

def draw_scene_geometry(screen):
    """绘制背景中的物理世界（模仿图片中的 Catapult 关卡）"""
    # 1. 地面 (绿色)
    pygame.draw.rect(screen, GROUND_COLOR, (0, 500, 150, 100)) # 左底座
    pygame.draw.rect(screen, GROUND_COLOR, (450, 500, 150, 100)) # 右底座
    
    # 2. 黑色障碍物 (Black Objects)
    pygame.draw.rect(screen, BLOCK_COLOR, (50, 400, 10, 100)) # 左支柱
    pygame.draw.rect(screen, BLOCK_COLOR, (130, 400, 10, 100)) # 中支柱
    pygame.draw.rect(screen, BLOCK_COLOR, (350, 400, 10, 100)) # 右侧障碍
    pygame.draw.rect(screen, BLOCK_COLOR, (40, 400, 320, 10)) # 横梁

    # 3. 蓝色平台 (Blue Platform)
    pygame.draw.line(screen, PLATFORM_COLOR, (40, 390), (400, 390), 5)

    # 4. 红球 (Target)
    pygame.draw.circle(screen, RED_BALL_COLOR, (60, 380), 10)

def generate_mock_data(center, count, spread):
    """生成模拟的采样点数据 (高斯分布)"""
    points = []
    for _ in range(count):
        x = int(random.gauss(center[0], spread))
        y = int(random.gauss(center[1], spread * 2)) # Y轴拉长一点模仿图中的分布
        points.append((x, y))
    return points

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Virtual Tools Visualization Style")

    # --- 1. 预处理资源 ---
    # 预先生成三种颜色的“光晕”笔刷，半径设为 30px
    blob_radius = 30
    brush_pink = create_soft_blob(COLOR_PINK, blob_radius, max_alpha=30)
    brush_cyan = create_soft_blob(COLOR_CYAN, blob_radius, max_alpha=30)
    brush_yellow = create_soft_blob(COLOR_YELLOW, blob_radius, max_alpha=30)

    # --- 2. 生成模拟数据 ---
    # 模仿图中左侧粉色聚集，中间混合，右侧黄色聚集的效果
    data_pink = generate_mock_data((100, 200), 150, 20)
    data_cyan = generate_mock_data((180, 220), 150, 25)
    data_yellow = generate_mock_data((260, 200), 150, 20)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 3. 绘图流程 ---
        
        # A. 清屏
        screen.fill(BG_COLOR)

        # B. 绘制热力图层 (Heatmap Layer)
        # 这一层在物体后面还是前面取决于需求，图中看起来是在物体上层但透明
        
        # 混合模式：这里使用简单的 Alpha Blending 即可
        # 遍历所有数据点，绘制“光晕”
        for p in data_pink:
            screen.blit(brush_pink, (p[0] - blob_radius, p[1] - blob_radius))
        for p in data_cyan:
            screen.blit(brush_cyan, (p[0] - blob_radius, p[1] - blob_radius))
        for p in data_yellow:
            screen.blit(brush_yellow, (p[0] - blob_radius, p[1] - blob_radius))

        # C. 绘制场景几何体 (覆盖在热力图下方或上方，图中几何体很清晰，建议放在热力图上方)
        draw_scene_geometry(screen)

        # D. 绘制具体的采样点 (Scatter Points)
        # 图中还有黑色/深色的小点代表具体的 sample
        for p in data_pink:
            pygame.draw.circle(screen, (100, 0, 100), p, 2)
        for p in data_cyan:
            pygame.draw.circle(screen, (0, 100, 100), p, 2)
        for p in data_yellow:
            pygame.draw.circle(screen, (100, 100, 0), p, 2)
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()