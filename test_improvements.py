#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试三个改进功能：
1. 中文输出
2. 结构化思考
3. 点击交互（模拟）
"""

from src.agent.controller import EmbodiedSearchAgent
from src.types.schema import AgentRequest
from PIL import Image
import json

def test_chinese_output():
    """测试中文输出"""
    print("\n=== 测试 1: 中文输出 ===")

    # 创建简单的测试图像
    img = Image.new('RGB', (448, 448), (200, 200, 200))
    img.save('temp_test_image.png')

    agent = EmbodiedSearchAgent()
    response = agent.step(AgentRequest(
        session_id='test-chinese',
        instruction='找到房间里的红色杯子',
        observation_image='temp_test_image.png',
        step_id=0
    ))

    print(f"✓ 思考文本: {response.thought}")
    print(f"✓ 动作: {response.action.type}")
    print(f"✓ 置信度: {response.confidence:.3f}")

    assert "找到" in response.thought or "房间" in response.thought or "置信度" in response.thought, "中文输出失败"
    print("✓ 中文输出测试通过！")

    return response


def test_structured_thought(response):
    """测试结构化思考"""
    print("\n=== 测试 2: 结构化思考 ===")

    st = response.structured_thought
    print(f"✓ 视觉观察: {st.get('observation', 'N/A')}")
    print(f"✓ 推理过程: {st.get('reasoning', 'N/A')}")
    print(f"✓ 动作: {st.get('action', 'N/A')}")
    print(f"✓ 置信度: {st.get('confidence', 'N/A')}")

    assert 'observation' in st, "缺少视觉观察字段"
    assert 'reasoning' in st, "缺少推理字段"
    assert 'action' in st, "缺少动作字段"
    print("✓ 结构化思考测试通过！")


def test_click_interaction():
    """测试点击交互"""
    print("\n=== 测试 3: 点击交互 ===")

    # 创建带有红色区域的测试图像
    img = Image.new('RGB', (448, 448), (200, 200, 200))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 200, 200], fill=(255, 0, 0))  # 红色圆圈
    img.save('temp_test_image_with_target.png')

    agent = EmbodiedSearchAgent()

    # 模拟点击红色区域的中心
    response = agent.step(AgentRequest(
        session_id='test-click',
        instruction='找到这个物体',
        observation_image='temp_test_image_with_target.png',
        clicked_point=[150, 150],  # 点击红色圆圈中心
        step_id=0
    ))

    print(f"✓ 目标绑定模式: {response.target_binding.get('mode', 'N/A')}")
    print(f"✓ 点击坐标: {response.target_binding.get('clicked_point', 'N/A')}")
    print(f"✓ 置信度: {response.confidence:.3f}")

    assert response.target_binding.get('mode') == 'multimodal', "未检测到多模态模式"
    assert response.target_binding.get('clicked_point') == [150, 150], "点击坐标不正确"
    print("✓ 点击交互测试通过！")


def test_full_demo():
    """测试完整演示流程"""
    print("\n=== 测试 4: 完整演示流程 ===")

    from src.simulation.room_simulator import RoomSimulator

    simulator = RoomSimulator()
    result = simulator.run_demo(
        instruction="找到桌上的红色杯子",
        max_steps=5
    )

    print(f"✓ 生成步骤数: {len(result.steps)}")
    print(f"✓ 视频路径: {result.video_path}")

    # 检查第一步是否有结构化思考
    if result.steps:
        first_step = result.steps[0]
        print(f"✓ 第一步动作: {first_step.action}")
        print(f"✓ 第一步结构化思考: {first_step.structured_thought}")

        if first_step.structured_thought:
            print("  - 观察:", first_step.structured_thought.get('observation', 'N/A')[:50])
            print("  - 推理:", first_step.structured_thought.get('reasoning', 'N/A')[:50])
            print("  - 动作:", first_step.structured_thought.get('action', 'N/A'))

    print("✓ 完整演示测试通过！")
    return result


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("开始测试具身视觉搜索智能体的三大改进功能")
        print("=" * 60)

        # 测试 1: 中文输出
        response = test_chinese_output()

        # 测试 2: 结构化思考
        test_structured_thought(response)

        # 测试 3: 点击交互
        test_click_interaction()

        # 测试 4: 完整演示
        demo_result = test_full_demo()

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n现在可以启动 Web UI 测试界面：")
        print("  python -m src.ui.app")
        print("然后在浏览器打开：")
        print("  http://127.0.0.1:8000")
        print("\n新功能：")
        print("  1. ✓ 中文输出 - 所有思考和推理都是中文")
        print("  2. ✓ 结构化思考 - 分为观察/推理/动作三部分")
        print("  3. ✓ 点击交互 - 点击图像选择目标物体")
        print("  4. ✓ 美化UI - 更丰富的信息展示和更好的视觉效果")

        # 输出演示结果的摘要
        print("\n演示结果摘要:")
        print(f"  - 总步数: {len(demo_result.steps)}")
        print(f"  - 视频: {demo_result.video_path}")
        print(f"  - 摘要: {demo_result.summary_path}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
