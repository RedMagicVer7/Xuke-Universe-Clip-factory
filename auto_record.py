"""
Xuke Clip Factory — OBS 自动录制模块
通过 OBS WebSocket 自动化直播录制
支持定时录制和手动触发
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# OBS WebSocket 客户端
# ──────────────────────────────────────────────

class OBSController:
    """OBS WebSocket 控制器"""

    def __init__(self, host: str = "localhost", port: int = 4455,
                 password: str = ""):
        self.host = host
        self.port = port
        self.password = password
        self._ws = None

    def connect(self):
        """连接 OBS WebSocket"""
        try:
            import obsws_python as obs
            self._ws = obs.ReqClient(
                host=self.host,
                port=self.port,
                password=self.password if self.password else None,
            )
            logger.info(f"已连接 OBS WebSocket: {self.host}:{self.port}")
            return True
        except ImportError:
            logger.error(
                "obsws-python 未安装。请运行: pip install obsws-python\n"
                "同时确保 OBS 已启用 WebSocket 服务器 (工具 → WebSocket 服务器设置)"
            )
            return False
        except Exception as e:
            logger.error(f"OBS 连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._ws:
            try:
                self._ws = None
            except Exception:
                pass

    def is_connected(self) -> bool:
        """检查是否连接"""
        if self._ws is None:
            return False
        try:
            self._ws.get_version()
            return True
        except Exception:
            return False

    def start_recording(self) -> bool:
        """开始录制"""
        if not self._ws:
            logger.error("未连接 OBS")
            return False
        try:
            self._ws.start_record()
            logger.info("OBS 开始录制")
            return True
        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            return False

    def stop_recording(self) -> Optional[str]:
        """停止录制，返回录制文件路径"""
        if not self._ws:
            return None
        try:
            result = self._ws.stop_record()
            output_path = result.output_path if hasattr(result, 'output_path') else ""
            logger.info(f"OBS 停止录制: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            return None

    def is_recording(self) -> bool:
        """检查是否正在录制"""
        if not self._ws:
            return False
        try:
            status = self._ws.get_record_status()
            return status.output_active
        except Exception:
            return False

    def set_output_directory(self, directory: str):
        """设置录制输出目录"""
        if not self._ws:
            return
        try:
            # OBS 5.x: 通过 Profile 设置
            self._ws.set_profile_parameter(
                "Output", "FilePath", directory
            )
            logger.info(f"录制输出目录: {directory}")
        except Exception as e:
            logger.warning(f"设置输出目录失败（可能需要手动设置）: {e}")

    def get_recording_status(self) -> dict:
        """获取当前录制状态"""
        if not self._ws:
            return {"active": False, "duration": 0}
        try:
            status = self._ws.get_record_status()
            return {
                "active": status.output_active,
                "duration": getattr(status, "output_duration", 0),
                "bytes": getattr(status, "output_bytes", 0),
            }
        except Exception:
            return {"active": False, "duration": 0}


# ──────────────────────────────────────────────
# 定时录制调度器
# ──────────────────────────────────────────────

class RecordingScheduler:
    """
    定时录制调度
    支持设定每天的录制时间段（如徐可直播通常在晚上 8-11 点）
    """

    def __init__(self, obs_controller: OBSController, config: dict):
        self.obs = obs_controller
        self.config = config
        self._timer = None
        self._is_running = False
        self._on_recording_done: Optional[Callable] = None

    def set_callback(self, callback: Callable):
        """设置录制完成后的回调（用于自动触发 pipeline）"""
        self._on_recording_done = callback

    def schedule_daily(self, start_hour: int = 20, start_minute: int = 0,
                       duration_hours: float = 3.5):
        """
        设置每日定时录制
        
        Args:
            start_hour: 开始录制的小时（24h制）
            start_minute: 开始录制的分钟
            duration_hours: 录制时长（小时）
        """
        now = datetime.now()
        target = now.replace(hour=start_hour, minute=start_minute,
                             second=0, microsecond=0)

        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(
            f"定时录制已设置: {target.strftime('%Y-%m-%d %H:%M')}, "
            f"时长 {duration_hours}h, "
            f"距离开始 {wait_seconds/3600:.1f}h"
        )

        self._timer = threading.Timer(
            wait_seconds,
            self._start_timed_recording,
            args=[duration_hours]
        )
        self._timer.daemon = True
        self._timer.start()
        self._is_running = True

    def _start_timed_recording(self, duration_hours: float):
        """执行定时录制"""
        logger.info(f"定时录制开始! 将录制 {duration_hours} 小时")

        if not self.obs.is_connected():
            if not self.obs.connect():
                logger.error("无法连接 OBS，定时录制失败")
                return

        if self.obs.start_recording():
            duration_seconds = duration_hours * 3600

            # 等待录制完成
            time.sleep(duration_seconds)

            output_path = self.obs.stop_recording()
            logger.info(f"定时录制完成: {output_path}")

            # 触发回调
            if self._on_recording_done and output_path:
                self._on_recording_done(output_path)

            # 设置明天的录制
            self.schedule_daily(
                start_hour=datetime.now().hour,
                duration_hours=duration_hours
            )

    def manual_start(self) -> bool:
        """手动开始录制"""
        if not self.obs.is_connected():
            if not self.obs.connect():
                return False
        return self.obs.start_recording()

    def manual_stop(self) -> Optional[str]:
        """手动停止录制"""
        output_path = self.obs.stop_recording()
        if self._on_recording_done and output_path:
            self._on_recording_done(output_path)
        return output_path

    def cancel(self):
        """取消定时录制"""
        if self._timer:
            self._timer.cancel()
            self._is_running = False
            logger.info("定时录制已取消")


# ──────────────────────────────────────────────
# 便捷函数
# ──────────────────────────────────────────────

def create_obs_controller(config: dict) -> OBSController:
    """从配置创建 OBS 控制器"""
    obs_cfg = config.get("obs", {})
    return OBSController(
        host=obs_cfg.get("host", "localhost"),
        port=obs_cfg.get("port", 4455),
        password=obs_cfg.get("password", ""),
    )


def quick_record(config: dict, duration_minutes: int = 0) -> Optional[str]:
    """
    快捷录制
    
    Args:
        config: 配置
        duration_minutes: 录制时长（分钟）, 0=手动停止
    
    Returns:
        录制文件路径
    """
    obs = create_obs_controller(config)
    if not obs.connect():
        return None

    obs.start_recording()
    print("录制中... ", end="", flush=True)

    if duration_minutes > 0:
        for remaining in range(duration_minutes * 60, 0, -10):
            time.sleep(10)
            if remaining % 60 == 0:
                print(f"{remaining//60}min ", end="", flush=True)
    else:
        input("按 Enter 停止录制...")

    output_path = obs.stop_recording()
    obs.disconnect()
    print(f"\n录制完成: {output_path}")
    return output_path


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    if not os.path.exists(config_file):
        print(f"配置文件不存在: {config_file}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=== OBS 自动录制模块 ===")
    print("1. 测试连接")
    print("2. 手动录制")
    print("3. 定时录制（每天 20:00）")

    choice = input("\n选择: ").strip()

    obs = create_obs_controller(cfg)

    if choice == "1":
        if obs.connect():
            print("OBS 连接成功!")
            status = obs.get_recording_status()
            print(f"录制状态: {'录制中' if status['active'] else '待机'}")
            obs.disconnect()
        else:
            print("连接失败 — 请确保 OBS 已启动并开启 WebSocket 服务器")

    elif choice == "2":
        path = quick_record(cfg)
        if path:
            print(f"文件保存在: {path}")

    elif choice == "3":
        scheduler = RecordingScheduler(obs, cfg)
        scheduler.schedule_daily(start_hour=20, duration_hours=3.5)
        print("定时录制已设置，程序将持续运行...")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            scheduler.cancel()
            print("\n已退出")
