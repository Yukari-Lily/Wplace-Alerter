import os 
import re
import io
import cv2
import sys
import time
import base64
import queue
import signal
import hashlib
import threading
import datetime
import requests
import numpy as np
import yaml
import logging
from typing import List, Dict, Tuple, Optional, Set
from PIL import Image
from colorama import Fore, init
from flask import Flask, request, jsonify
from waitress import serve
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
logging.getLogger("urllib3").setLevel(logging.ERROR)

init(autoreset=True)

# ==========================
# 固定色板 + O(1) 查表
# ==========================
from enum import Enum

class Color(Enum):
    TRANSPARENT     = (0, 0, 0, 0)
    BLACK           = (0, 0, 0, 255)
    DARK_GRAY       = (60, 60, 60, 255)
    GRAY            = (120, 120, 120, 255)
    LIGHT_GRAY      = (210, 210, 210, 255)
    WHITE           = (255, 255, 255, 255)
    DEEP_RED        = (96, 0, 24, 255)
    RED             = (237, 28, 36, 255)
    ORANGE          = (255, 127, 39, 255)
    GOLD            = (246, 170, 9, 255)
    YELLOW          = (249, 221, 59, 255)
    LIGHT_YELLOW    = (255, 250, 188, 255)
    DARK_GREEN      = (14, 185, 104, 255)
    GREEN           = (19, 230, 123, 255)
    LIGHT_GREEN     = (135, 255, 94, 255)
    DARK_TEAL       = (12, 129, 110, 255)
    TEAL            = (16, 174, 166, 255)
    LIGHT_TEAL      = (19, 225, 190, 255)
    DARK_BLUE       = (40, 80, 158, 255)
    BLUE            = (64, 147, 228, 255)
    CYAN            = (96, 247, 242, 255)
    INDIGO          = (107, 80, 246, 255)
    LIGHT_INDIGO    = (153, 177, 251, 255)
    DARK_PURPLE     = (120, 12, 153, 255)
    PURPLE          = (170, 56, 185, 255)
    LIGHT_PURPLE    = (224, 159, 249, 255)
    DARK_PINK       = (203, 0, 122, 255)
    PINK            = (236, 31, 128, 255)
    LIGHT_PINK      = (243, 141, 169, 255)
    DARK_BROWN      = (104, 70, 52, 255)
    BROWN           = (149, 104, 42, 255)
    BEIGE           = (248, 178, 119, 255)
    MEDIUM_GRAY     = (170, 170, 170, 255)
    DARK_RED        = (165, 14, 30, 255)
    LIGHT_RED       = (250, 128, 114, 255)
    DARK_ORANGE     = (228, 92, 26, 255)
    LIGHT_TAN       = (214, 181, 148, 255)
    DARK_GOLDENROD  = (156, 132, 49, 255)
    GOLDENROD       = (197, 173, 49, 255)
    LIGHT_GOLDENROD = (232, 212, 95, 255)
    DARK_OLIVE      = (74, 107, 58, 255)
    OLIVE           = (90, 148, 74, 255)
    LIGHT_OLIVE     = (132, 197, 115, 255)
    DARK_CYAN       = (15, 121, 159, 255)
    LIGHT_CYAN      = (187, 250, 242, 255)
    LIGHT_BLUE      = (125, 199, 255, 255)
    DARK_INDIGO     = (77, 49, 184, 255)
    DARK_SLATE_BLUE = (74, 66, 132, 255)
    SLATE_BLUE      = (122, 113, 196, 255)
    LIGHT_SLATE_BLUE= (181, 174, 241, 255)
    LIGHT_BROWN     = (219, 164, 99, 255)
    DARK_BEIGE      = (209, 128, 81, 255)
    LIGHT_BEIGE     = (255, 197, 165, 255)
    DARK_PEACH      = (155, 82, 73, 255)
    PEACH           = (209, 128, 120, 255)
    LIGHT_PEACH     = (250, 182, 164, 255)
    DARK_TAN        = (123, 99, 82, 255)
    TAN             = (156, 132, 107, 255)
    DARK_SLATE      = (51, 57, 65, 255)
    SLATE           = (109, 117, 141, 255)
    LIGHT_SLATE     = (179, 185, 209, 255)
    DARK_STONE      = (109, 100, 63, 255)
    STONE           = (148, 140, 107, 255)
    LIGHT_STONE     = (205, 197, 158, 255)

COLOR_LIST = list(Color)

# 颜色->(name, idx) O(1) 查找
_PALETTE_MAP: Dict[Tuple[int, int, int, int], Tuple[str, int]] = {
    c.value: (c.name, i) for i, c in enumerate(COLOR_LIST)
}
def get_color_id(rgba) -> Tuple[Optional[str], Optional[int]]:
    return _PALETTE_MAP.get(tuple(rgba), (None, None))

# RGBA 打包为 uint32（便于 np.unique）
def _pack_rgba_u32_arr(rgba_arr: np.ndarray) -> np.ndarray:
    r = rgba_arr[..., 0].astype(np.uint32)
    g = rgba_arr[..., 1].astype(np.uint32)
    b = rgba_arr[..., 2].astype(np.uint32)
    a = rgba_arr[..., 3].astype(np.uint32)
    return (r << 24) | (g << 16) | (b << 8) | a

# 色板的 uint32 key 与 idx 映射
_PALETTE_KEY2IDX: Dict[int, int] = {
    ((r << 24) | (g << 16) | (b << 8) | a): i
    for i, (r, g, b, a) in enumerate(c.value for c in COLOR_LIST)
}

COLOR_NAME_MAP = {
    "TRANSPARENT": "透明", "BLACK": "黑色", "DARK_GRAY": "深灰色", "GRAY": "灰色", "LIGHT_GRAY": "浅灰色", "WHITE": "白色",
    "DEEP_RED": "深红色", "RED": "红色", "ORANGE": "橙色", "GOLD": "金色", "YELLOW": "黄色", "LIGHT_YELLOW": "浅黄色",
    "DARK_GREEN": "深绿色", "GREEN": "绿色", "LIGHT_GREEN": "浅绿色", "DARK_TEAL": "深青色", "TEAL": "青色", "LIGHT_TEAL": "浅青色",
    "DARK_BLUE": "深蓝色", "BLUE": "蓝色", "CYAN": "青蓝色", "INDIGO": "靛青", "LIGHT_INDIGO": "浅靛青",
    "DARK_PURPLE": "深紫色", "PURPLE": "紫色", "LIGHT_PURPLE": "浅紫色", "DARK_PINK": "深粉色", "PINK": "粉色", "LIGHT_PINK": "浅粉色",
    "DARK_BROWN": "深棕色", "BROWN": "棕色", "BEIGE": "米色", "MEDIUM_GRAY": "中灰色",
    "DARK_RED": "暗红色", "LIGHT_RED": "浅红色", "DARK_ORANGE": "深橙色", "LIGHT_TAN": "浅褐色",
    "DARK_GOLDENROD": "深金麒麟色", "GOLDENROD": "金麒麟色", "LIGHT_GOLDENROD": "浅金麒麟色",
    "DARK_OLIVE": "深橄榄色", "OLIVE": "橄榄色", "LIGHT_OLIVE": "浅橄榄色",
    "DARK_CYAN": "深青蓝色", "LIGHT_CYAN": "浅青蓝色", "LIGHT_BLUE": "浅蓝色",
    "DARK_INDIGO": "深靛青", "DARK_SLATE_BLUE": "深石板蓝", "SLATE_BLUE": "石板蓝", "LIGHT_SLATE_BLUE": "浅石板蓝",
    "LIGHT_BROWN": "浅棕色", "DARK_BEIGE": "深米色", "LIGHT_BEIGE": "浅米色", "DARK_PEACH": "深桃色", "PEACH": "桃色", "LIGHT_PEACH": "浅桃色",
    "DARK_TAN": "深褐色", "TAN": "褐色", "DARK_SLATE": "深石板灰", "SLATE": "石板灰", "LIGHT_SLATE": "浅石板灰",
    "DARK_STONE": "深石色", "STONE": "石色", "LIGHT_STONE": "浅石色",
}
def zh_color(name: Optional[str]) -> str:
    return COLOR_NAME_MAP.get(name, "未知颜色")

# ==========================
# 基础工具函数
# ==========================

def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def bytes_hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=16).hexdigest()

def save_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_bytes(path: str, byts: bytes) -> None:
    with open(path, "wb") as f:
        f.write(byts)

def read_bytes(path: str) -> Optional[bytes]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()

# ==========================
# 获取图片（支持 HTTPS 代理）
# ==========================

class ImageFetcher:

    def __init__(self, timeout: int = 10, https_proxy: Optional[str] = None, trust_env: bool = False):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = bool(trust_env)

        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if https_proxy:
            self.session.proxies = {"https": https_proxy}

    def fetch_png(self, url: str) -> bytes:
        r = self.session.get(url, timeout=self.timeout, headers={"Accept": "image/png,*/*;q=0.8"})
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "png" not in ctype and not url.lower().endswith(".png"):
            raise ValueError(f"响应 Content-Type 非 PNG：{ctype}")
        return r.content

# ==========================
# 差异引擎（索引图 + 索引哈希）
# ==========================

class DiffEngine:

    def __init__(self, max_detail_points: int = 10):
        self.max_detail_points = max_detail_points

    @staticmethod
    def crop_bytes_to_png(byts: bytes, crop_box: Tuple[int, int, int, int]) -> bytes:
        with Image.open(io.BytesIO(byts)) as im:
            x1, y1, x2, y2 = crop_box
            crop = im.crop((x1, y1, x2, y2))
            out = io.BytesIO()
            crop.save(out, format="PNG")
            return out.getvalue()

    @staticmethod
    def imread_rgba(byts: bytes) -> np.ndarray:
        arr = np.frombuffer(byts, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("图像解码失败")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif img.shape[2] == 4:
            img = img[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
        else:
            raise ValueError("不支持的通道数")
        return img

    @staticmethod
    def rgba_to_index_image(rgba: np.ndarray) -> np.ndarray:
        keys = _pack_rgba_u32_arr(rgba)
        flat = keys.reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        map_arr = np.full(uniq.shape, -1, dtype=np.int8)
        for j, k in enumerate(uniq):
            map_arr[j] = _PALETTE_KEY2IDX.get(int(k), -1)
        idx_flat = map_arr[inv]
        return idx_flat.reshape(keys.shape).astype(np.int8)

    def index_hash_from_rgba(self, rgba: np.ndarray) -> str:
        idx = self.rgba_to_index_image(rgba)
        # 直接对索引数组字节做 blake2b
        return hashlib.blake2b(idx.tobytes(), digest_size=16).hexdigest()

    def index_hash_from_png_bytes(self, byts: bytes) -> str:
        rgba = self.imread_rgba(byts)
        return self.index_hash_from_rgba(rgba)

    @staticmethod
    def same_shape(a: np.ndarray, b: np.ndarray) -> bool:
        return a.shape == b.shape

    @staticmethod
    def same_index(a_idx: np.ndarray, b_idx: np.ndarray) -> bool:
        return a_idx.shape == b_idx.shape and np.array_equal(a_idx, b_idx)

    @staticmethod
    def diff_regions_idx(a_idx: np.ndarray, b_idx: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if a_idx.shape != b_idx.shape:
            raise ValueError("尺寸不一致，无法比较")
        mask = (a_idx != b_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours]

    def changed_pixels_from_idx(
        self, a_idx: np.ndarray, b_idx: np.ndarray, limit: Optional[int] = None
    ) -> Tuple[List[Dict[str, object]], int]:
        if a_idx.shape != b_idx.shape:
            raise ValueError("尺寸不一致，无法逐像素比较")
        mask = (a_idx != b_idx)
        ys, xs = np.nonzero(mask)
        total = len(xs)
        if limit is not None and total > limit:
            idxs = np.linspace(0, total - 1, limit, dtype=int)
            xs = xs[idxs]; ys = ys[idxs]
        changed: List[Dict[str, object]] = []
        for x, y in zip(xs, ys):
            old_i = int(a_idx[y, x]); new_i = int(b_idx[y, x])
            if 0 <= new_i < len(COLOR_LIST):
                nr, ng, nb, na = COLOR_LIST[new_i].value
            else:
                nr, ng, nb, na = (0, 0, 0, 0)
            if 0 <= old_i < len(COLOR_LIST):
                or_, og, ob, oa = COLOR_LIST[old_i].value
            else:
                or_, og, ob, oa = (0, 0, 0, 0)
            changed.append({
                "x": int(x),
                "y": int(y),
                "new_color": (int(nr), int(ng), int(nb), int(na)),
                "old_color": (int(or_), int(og), int(ob), int(oa)),
                "old_idx": old_i,
                "new_idx": new_i,
            })
        return changed, total

# ==========================
# 通知（OneBot v11 / Napcat）
# ==========================

class Notifier:
    def __init__(self, napcat_cfg: Dict):
        self.napcat = napcat_cfg or {}

    def send(self, message: str, image_path1: Optional[str] = None, image_path2: Optional[str] = None) -> None:
        napcat = self.napcat
        base_url = napcat.get("base_url", "")
        access_token = napcat.get("access_token", "")
        target_type = napcat.get("target_type", "group")
        target_id = napcat.get("id")
        if not base_url or not target_id:
            print(Fore.LIGHTYELLOW_EX + "[Notifier] Napcat 未配置，跳过发送")
            return

        if target_type == "private":
            endpoint = f"{base_url}/send_private_msg"
            payload_core = {"user_id": int(target_id)}
        else:
            endpoint = f"{base_url}/send_group_msg"
            payload_core = {"group_id": int(target_id)}

        text_concat = (message).strip()

        def split_text(txt: str, maxlen: int = 1500) -> List[str]:
            return [txt[i:i+maxlen] for i in range(0, len(txt), maxlen)] or [""]

        segments = [{"type": "text", "data": {"text": chunk}} for chunk in split_text(text_concat)]

        def encode_image_to_base64(p: Optional[str]) -> Optional[str]:
            if not p or not os.path.exists(p):
                return None
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"base64://{b64}"

        for p in [image_path1, image_path2]:
            b64 = encode_image_to_base64(p)
            if b64:
                segments.append({"type": "image", "data": {"file": b64}})

        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        try:
            payload = {**payload_core, "message": segments}
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            j = resp.json()
            if j.get("status") != "ok":
                print(Fore.LIGHTYELLOW_EX + f"Napcat 返回非 ok：{j}")
        except requests.exceptions.HTTPError as errh:
            print(Fore.LIGHTRED_EX + f"HTTP 错误：{errh}")
        except requests.exceptions.ConnectionError as errc:
            print(Fore.LIGHTRED_EX + f"连接错误：{errc}")
        except requests.exceptions.Timeout as errt:
            print(Fore.LIGHTRED_EX + f"请求超时：{errt}")
        except requests.exceptions.RequestException as err:
            print(Fore.LIGHTRED_EX + f"请求异常：{err}")

# ==========================
# 业务编排（含“仅报警一次” + 索引哈希短路）
# ==========================

class Orchestrator:
    def __init__(self, fetcher: ImageFetcher, diff: DiffEngine, notifier: Notifier, settings: Dict):
        self.fetcher = fetcher
        self.diff = diff
        self.notifier = notifier
        self.settings = settings or {}
        self.cooldown = int(self.settings.get("cooldown", 30))
        self.alerted_once: Set[str] = set()

        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    @staticmethod
    def get_tiles_from_api_url(api_image: str) -> Tuple[int, int]:
        m = re.search(r"/files/s\d+/tiles/(\d+)/(\d+)\.png", api_image)
        if m:
            return int(m.group(1)), int(m.group(2))
        return (0, 0)

    def set_current_as_baseline(self, key: str) -> Dict[str, str]:
        lock = self._get_lock(key)
        with lock:
            arts = self.settings.get("arts", {})
            art = arts.get(key)
            if not art:
                return {"ok": False, "error": f"未找到 key={key} 的条目"}

            name = (art.get("name") or "").strip() or key
            api_image = art["api_image"]
            coords = (
                art["start_coords"]["x"],
                art["start_coords"]["y"],
                art["end_coords"]["x"],
                art["end_coords"]["y"],
            )
            path = f"data/{key}/"
            ensure_dir(path)

            try:
                # 拉取并裁剪当前最新图像
                png_bytes = self.fetcher.fetch_png(api_image)
                cropped_bytes = self.diff.crop_bytes_to_png(png_bytes, coords)
                # 计算索引哈希
                rgba = self.diff.imread_rgba(cropped_bytes)
                idx_hash = hashlib.blake2b(
                    self.diff.rgba_to_index_image(rgba).tobytes(), digest_size=16
                ).hexdigest()
            except Exception as e:
                return {"ok": False, "error": f"获取或解码图片失败：{e}"}

            # 覆盖写入 baseline 与当前
            save_bytes(f"{path}original.png", cropped_bytes)
            save_bytes(f"{path}good.png", cropped_bytes)
            save_bytes(f"{path}new.png", cropped_bytes)
            save_text(f"{path}good.hash", idx_hash)

            # 清除一次性报警状态
            if key in self.alerted_once:
                self.alerted_once.discard(key)

            print(Fore.LIGHTGREEN_EX + f"[baseline] {name} 已将当前最新图像设为原图。")
            return {"ok": True, "key": key, "message": f"{name} 已重设为基准"}

    def refresh_one(self, key: str) -> Dict[str, str]:
        lock = self._get_lock(key)
        with lock:
            arts = self.settings.get("arts", {})
            art = arts.get(key)
            if not art:
                return {"status": "error", "message": f"未找到 key={key} 的条目"}
            if not art.get("track"):
                return {"status": "skipped", "message": f"key={key} 未开启 track"}

            name = (art.get("name") or "").strip() or key
            api_image = art["api_image"]
            coords = (
                art["start_coords"]["x"],
                art["start_coords"]["y"],
                art["end_coords"]["x"],
                art["end_coords"]["y"],
            )
            path = f"data/{key}/"
            ensure_dir(path)

            # 1) 下载并裁剪
            png_bytes = self.fetcher.fetch_png(api_image)
            cropped_bytes = self.diff.crop_bytes_to_png(png_bytes, coords)

            # ---- 1.2) 索引哈希短路----
            hash_path = f"{path}good.hash"
            new_rgba = self.diff.imread_rgba(cropped_bytes)
            new_idx  = self.diff.rgba_to_index_image(new_rgba)
            new_idx_hash = hashlib.blake2b(new_idx.tobytes(), digest_size=16).hexdigest()
            old_idx_hash = read_text(hash_path)

            # 若已有索引哈希且一致：直接判为无变化
            if old_idx_hash and new_idx_hash == old_idx_hash:
                if key in self.alerted_once:
                    self.alerted_once.discard(key)
                save_bytes(f"{path}new.png", cropped_bytes)
                print(Fore.LIGHTGREEN_EX + "未检测到像素变化。（hash检测）")
                return {"status": "nochange", "message": f"未变化：{name}（hash检测）"}

            # 1.5) 二进制哈希短路
            old_good_bytes = read_bytes(f"{path}good.png")
            if old_good_bytes is not None:
                if bytes_hash(cropped_bytes) == bytes_hash(old_good_bytes):
                    # 同时补写/校正索引哈希文件
                    try:
                        good_idx_hash = self.diff.index_hash_from_png_bytes(old_good_bytes)
                        save_text(hash_path, good_idx_hash)
                    except Exception:
                        pass
                    if key in self.alerted_once:
                        self.alerted_once.discard(key)
                    save_bytes(f"{path}new.png", cropped_bytes)
                    print(Fore.LIGHTGREEN_EX + "未检测到像素变化。（hash检测）")
                    return {"status": "nochange", "message": f"未变化：{name}（hash检测）"}

            # 2) 保存 new
            save_bytes(f"{path}new.png", cropped_bytes)

            # 初次：建立基准（写 good.png + good.hash）
            if old_good_bytes is None:
                if key in self.alerted_once:
                    self.alerted_once.discard(key)
                print(Fore.LIGHTYELLOW_EX + "未找到基准图像，已将新图像保存为基准。")
                save_bytes(f"{path}original.png", cropped_bytes)
                save_bytes(f"{path}good.png", cropped_bytes)
                save_text(hash_path, new_idx_hash)
                return {"status": "baseline", "message": f"初始化基准：{name}"}

            # 3) RGBA -> 索引图，做严格比较
            good_rgba = self.diff.imread_rgba(old_good_bytes)
            if not self.diff.same_shape(good_rgba, new_rgba):
                print(Fore.LIGHTRED_EX + f"错误：图像尺寸不一致。good={good_rgba.shape}, new={new_rgba.shape}")
                return {"status": "error", "message": f"尺寸不一致：{name}"}

            good_idx = self.diff.rgba_to_index_image(good_rgba)

            # original 判断是否恢复
            restored = False
            orig_path = f"{path}original.png"
            if os.path.exists(orig_path):
                ob = read_bytes(orig_path)
                if ob is not None:
                    orig_rgba = self.diff.imread_rgba(ob)
                    if self.diff.same_shape(orig_rgba, new_rgba):
                        orig_idx = self.diff.rgba_to_index_image(orig_rgba)
                        restored = self.diff.same_index(orig_idx, new_idx)

            # 4) 索引图严格判定
            if self.diff.same_index(good_idx, new_idx):
                if key in self.alerted_once:
                    self.alerted_once.discard(key)
                try:
                    save_text(hash_path, new_idx_hash)
                except Exception:
                    pass
                print(Fore.LIGHTGREEN_EX + "未检测到像素变化。（像素判定）")
                return {"status": "nochange", "message": f"未变化：{name}（像素判定）"}

            if restored:
                if key in self.alerted_once:
                    self.alerted_once.discard(key)
                print(Fore.LIGHTGREEN_EX + "图像已被恢复。")
                save_bytes(f"{path}good.png", cropped_bytes)
                save_text(hash_path, new_idx_hash)
                return {"status": "restored", "message": f"已恢复：{name}"}

            # 5) 差异区域 + 像素明细
            bboxes = self.diff.diff_regions_idx(good_idx, new_idx)
            sample_limit = max(1, min(5000, max(self.diff.max_detail_points, 20)))
            changed, total = self.diff.changed_pixels_from_idx(good_idx, new_idx, limit=sample_limit)

            # 详情文本
            details: List[str] = []
            topN = min(self.diff.max_detail_points, len(changed))
            for i in range(topN):
                px = changed[i]
                old_idx = px["old_idx"]; new_idx_i = px["new_idx"]
                old_name = COLOR_LIST[old_idx].name if 0 <= old_idx < len(COLOR_LIST) else None
                new_name = COLOR_LIST[new_idx_i].name if 0 <= new_idx_i < len(COLOR_LIST) else None
                line = f"[{i+1}] X={coords[0] + px['x']}, Y={coords[1] + px['y']} {zh_color(old_name)} → {zh_color(new_name)}"
                details.append(line)
            if total > topN:
                details.append(f"...(共 {total} 个像素修改，{len(bboxes)} 个区域)")

            name_fmt = f"[{name}]" if name else ""
            msg = f"[Wplace]{name_fmt} 检测到 {total} 个像素被修改喵！\n修改详情：\n" + ("\n".join(details) if details else "（变化较大，略去具体像素）")

            # ——仅报警一次逻辑——
            if key not in self.alerted_once:
                self.notifier.send(
                    message=msg,
                    image_path1=f"{path}new.png",
                    image_path2=f"{path}good.png",
                )
                self.alerted_once.add(key)
                print(Fore.LIGHTRED_EX + f"检测到 {total} 个像素被修改喵！")
            else:
                print(Fore.LIGHTYELLOW_EX + f"图像有修改并且未修复（{name}，变更像素 {total}）。")

            return {"status": "changed", "message": f"检测到变化：{name}", "changed": total, "regions": len(bboxes)}

# ==========================
# 串行任务队列（带去重）
# ==========================

class RefreshWorker:
    def __init__(self, orchestrator: Orchestrator):
        self.q: "queue.Queue[str]" = queue.Queue()
        self.orch = orchestrator
        self._stop = threading.Event()
        self.in_flight: Set[str] = set()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def submit(self, key: str):
        if key in self.in_flight:
            return
        self.in_flight.add(key)
        self.q.put(key)

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                key = self.q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self.orch.refresh_one(key)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(Fore.LIGHTRED_EX + f"[Worker] 刷新 {key} 出错：{e}")
            finally:
                self.in_flight.discard(key)
                self.q.task_done()

# ==========================
# HTTP 服务
# ==========================

def create_http_app(worker: RefreshWorker, settings: Dict) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    def _do_refresh(all_flag: bool = False, key: Optional[str] = None):
        results = []
        if all_flag:
            for k, art in (settings.get("arts") or {}).items():
                if not art.get("track"):
                    continue
                res = worker.orch.set_current_as_baseline(k)
                results.append({"key": k, **res})
            return {"ok": True, "results": results}, 200
        elif key:
            res = worker.orch.set_current_as_baseline(key)
            return {"ok": True, "result": {"key": key, **res}}, 200
        else:
            return {"ok": False, "error": "缺少参数"}, 400

    # 新增 GET：刷新全部
    @app.route("/refresh/all", methods=["GET"])
    def http_refresh_all():
        payload, status = _do_refresh(all_flag=True)
        return jsonify(payload), status

    # 新增 GET：刷新单个
    @app.route("/refresh/<key>", methods=["GET"])
    def http_refresh_one(key: str):
        payload, status = _do_refresh(key=key)
        return jsonify(payload), status

    return app

# ==========================
# 主程序
# ==========================

def _get_https_proxy_from_cfg(cfg: dict) -> Optional[str]:
    p = cfg.get("https_proxy")
    if p:
        return str(p).strip() or None
    proxy_obj = cfg.get("proxy") or {}
    p2 = proxy_obj.get("https")
    return (str(p2).strip() or None) if p2 else None

def main(arts_data: dict):
    https_proxy = _get_https_proxy_from_cfg(arts_data)
    fetcher = ImageFetcher(timeout=10, https_proxy=https_proxy, trust_env=False)
    diff = DiffEngine(max_detail_points=int(arts_data.get("diff", {}).get("max_points", 10)))
    notifier = Notifier(arts_data.get("napcat", {}))
    orch = Orchestrator(fetcher, diff, notifier, arts_data)
    worker = RefreshWorker(orch)

    def cleanup(*_):
        print("\n检测到 Ctrl+C，中止程序...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    app = create_http_app(worker, arts_data)
    port = int(arts_data.get("http_port", 8000))
    print(f"{now_str()} HTTP 已启动：127.0.0.1:{port}")

    def http_serve():
        serve(app, host="127.0.0.1", port=port)

    http_thread = threading.Thread(target=http_serve, daemon=True)
    http_thread.start()

    while True:
        print(now_str())
        try:
            for key, art in (arts_data.get("arts") or {}).items():
                if not art.get("track"):
                    continue
                name = (art.get("name") or "").strip() or key
                print(Fore.LIGHTYELLOW_EX + f"正在检查: {Fore.RESET}{name}", end=" -> ")
                worker.submit(key)
                time.sleep(5)
        except KeyboardInterrupt:
            print("检测到 Ctrl+C，中止程序...")
            break
        except Exception as e:
            print(f"错误：{e}")
        time.sleep(int(arts_data.get("cooldown", 30)))
        print()

def load_config_yaml() -> dict:
    for p in ("data/arts.yaml", "data/arts.yml"):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{p} 解析后不是字典对象，请检查 YAML 格式")
            return data
    raise FileNotFoundError("未找到 YAML 配置（data/arts.yaml 或 data/arts.yml）")

if __name__ == "__main__":
    arts_data = load_config_yaml()
    main(arts_data)
