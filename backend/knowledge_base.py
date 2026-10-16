# -*- coding: utf-8 -*-
"""
RAG 知识库 —— 物品百科 + 任务需求 + 藏身处配方 → 向量库 → 语义检索

流程：
  1. 构建文档：把物品百科、任务需求、藏身处配方/制造/换物转成中文文本 chunk
  2. 向量化：用 fastembed(bge-small-zh-v1.5) 把所有文档 embedding 成向量
  3. 检索：用户问句 embedding → 余弦相似度 → top-k 返回
  4. 缓存：向量和文档构建一次后存本地（json + npy），下次秒加载

Embedding 模型：BAAI/bge-small-zh-v1.5（中文检索优化，512 维，ONNX 运行时，无需 torch）
"""
import os
import json
import re
import threading
from pathlib import Path

import numpy as np

from tarkov_client import tarkov

# 国内网络加速：HuggingFace 模型下载走镜像（必须在 import fastembed 前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
# 关闭 Windows 上 symlink 缓存警告（模型已本地化到项目内，无需联网）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 模型本地缓存目录：backend/models（而非系统 Temp，保证持久、免重复下载）
MODEL_DIR = str(Path(__file__).resolve().parent / "models")

# 缓存目录与文件
CACHE_DIR = Path(__file__).resolve().parent / "data" / "kb_cache"
DOCS_FILE = CACHE_DIR / "docs.json"
VECS_FILE = CACHE_DIR / "vectors.npy"

# bge 模型官方推荐的 query 前缀（只加在查询侧，提升短查询检索效果）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 物品类型中文映射
TYPE_MAP = {
    "gun": "武器", "ammo": "弹药", "armor": "护甲", "barter": "物资",
    "wearable": "穿戴", "container": "容器", "key": "钥匙",
}

# ===== 词法加权（解决短查询/任务名查询漂移）=====
# 语义检索对短查询不稳：问「收藏家」三个字，向量会漂到「藏身处」「制造配方」上去，
# 目标文档直接掉出 top-5，模型于是回答「知识库没有收录这一项」。
# 这里补一层词法匹配：查询与文档标题（meta.name / normalizedName）的重合度，
# 叠加到语义分数上排序。词法信号只用于「排得更靠前」，不参与 score 上报。
LEXICAL_WEIGHT = 0.8
TITLE_FIELDS = ("name", "normalizedName")
# 标题「原样出现在查询里」才算强信号，且必须有足够长度。
# 中文取 3 字：「收藏家」能命中，而「什么」「任务」这类泛词不会把无关文档顶上来
# （实测「显卡有什么用」曾因匹配到「什么」把无关任务顶到第一）。
# 英文取 5 字符：让 collector / ledx-skin-transilluminator 这类规范名能命中，
# 而 ledx、gpu 这种短缩写不参与（避免泛命中）。
MIN_TITLE_SUBSTR_CJK = 3
MIN_TITLE_SUBSTR_ASCII = 5

# 汉语高频功能词组合：这类 n-gram 命中不代表「用户点名的就是这个文档」，
# 参与打分只会在语义分接近时把无关文档顶上来（实测「显卡有什么用」会被
# 「U盘里有什么？」靠「有什么」抢到第一）。从 n-gram 集合里剔掉。
_NOISE_SUBSTRINGS = (
    "什么", "怎么", "哪些", "哪个", "多少", "多少", "需要", "有什", "有哪",
    "可以", "能否", "如何", "为什", "这个", "那个", "一下", "谢谢", "请问",
)


def _is_noise(gram: str) -> bool:
    return any(g in gram for g in _NOISE_SUBSTRINGS)


def _substring_eligible(title: str) -> bool:
    """标题长度是否够格作为「被查询点名」的强信号。"""
    if title.isascii():
        return len(title) >= MIN_TITLE_SUBSTR_ASCII
    return len(title) >= MIN_TITLE_SUBSTR_CJK


def _char_ngrams(text: str, lo: int = 2, hi: int = 4) -> set:
    """中文按 2~4 字滑窗，英文/数字按整词，全部小写。

    高频功能词组合（见 _NOISE_SUBSTRINGS）不参与，避免「有什么」这种
    泛化重合把无关文档加权上来。
    """
    if not text:
        return set()
    grams = set()
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        for n in range(lo, hi + 1):
            for i in range(len(run) - n + 1):
                gram = run[i:i + n]
                if not _is_noise(gram):
                    grams.add(gram)
    for tok in re.findall(r'[A-Za-z0-9]+', text):
        grams.add(tok.lower())
    return grams


def _title_lexical_score(query: str, meta: dict) -> float:
    """查询与文档标题的词法重合度（0~1），用于给精确提及的文档加权。"""
    titles = [str(meta.get(f, "")) for f in TITLE_FIELDS]
    titles = [t for t in titles if t]
    if not query or not titles:
        return 0.0

    q_lower = query.lower()
    q_grams = _char_ngrams(query)
    q_len = len(q_grams) or 1

    best = 0.0
    for title in titles:
        t_lower = title.lower()
        score = 0.0
        # 标题被查询完整包含（「收藏家任务要交什么」含「收藏家」/「Collector 任务」含「collector」）
        if _substring_eligible(title) and t_lower in q_lower:
            score = 1.0
        else:
            # 两个方向的 n-gram 覆盖率取大者（已剔除高频功能词组合）
            t_grams = _char_ngrams(title)
            t_len = len(t_grams) or 1
            score = max(len(q_grams & t_grams) / q_len,
                        len(q_grams & t_grams) / t_len)
            # 弱重合不给加权：语义分本身很接近时，一点点词法噪音就能改变排序，
            # 所以只有「明显点名」才算数（3 次方压平中低分段）。
            score = score ** 3
        best = max(best, score)
    return best

# 任务目标类型 → 动词（用于生成"需要上交/寻找 X"的描述）
OBJECTIVE_VERB = {
    "giveItem": "上交", "findItem": "寻找", "plantItem": "放置",
    "findQuestItem": "寻找", "giveQuestItem": "上交", "useItem": "使用",
    "sellItem": "出售", "buildWeapon": "改装",
}


class KnowledgeBase:
    """向量知识库：负责文档构建、向量化、语义检索"""

    def __init__(self):
        self._model = None
        self.docs = []          # [{"text": str, "meta": dict}]
        self.vectors = None     # (N, dim) float32 矩阵
        self._built = False
        self._lock = threading.Lock()

    # ---------- 模型懒加载 ----------
    def _get_model(self):
        """懒加载 embedding 模型（模型缓存在项目内 models 目录，首次构建时下载）"""
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(
                model_name="BAAI/bge-small-zh-v1.5", cache_dir=MODEL_DIR)
        return self._model

    # ---------- 构建 ----------
    def build(self, force: bool = False):
        """构建知识库（带本地缓存）。force=True 时强制重建。"""
        with self._lock:
            if self._built:
                return
            if not force and self._load_cache():
                self._built = True
                return
            try:
                tarkov.load()
                tarkov.load_knowledge()
                docs = self._build_docs()
                if not docs:
                    raise RuntimeError("知识库文档构建结果为空")
                model = self._get_model()
                texts = [d["text"] for d in docs]
                vectors = list(model.embed(texts, batch_size=64))
                self.docs = docs
                self.vectors = np.array(vectors, dtype=np.float32)
                self._built = True
                self._save_cache()
            except Exception as e:
                # 构建失败不抛给上层导致崩溃，记录并让检索返回空
                print(f"[知识库] 构建失败：{e}")
                self.docs = []
                self.vectors = None
                self._built = True  # 标记已尝试，避免反复重试

    # ---------- 文档构建 ----------
    def _build_docs(self):
        docs = []
        docs += self._item_docs()
        docs += self._task_docs()
        docs += self._craft_docs()
        docs += self._barter_docs()
        docs += self._hideout_docs()
        return docs

    def _item_docs(self):
        """物品百科：每个物品一段"""
        docs = []
        for iid, item in tarkov.items.items():
            name = tarkov.item_name(item)
            if not name:
                continue
            norm = item.get("normalizedName", "")
            type_zh = "/".join(TYPE_MAP.get(t, t) for t in item.get("types", [])) or "未知"
            w, h = item.get("width"), item.get("height")
            bp = item.get("basePrice")
            text = f"物品百科：{name}。英文名 {norm}，类型 {type_zh}，占格 {w}x{h}，基础价 {bp} 卢布。"
            docs.append({"text": text, "meta": {
                "type": "item", "id": iid, "name": name,
                "normalizedName": norm,
            }})
        return docs

    def _task_docs(self):
        """任务需求：每个任务一段，含所需物品中文名"""
        docs = []
        for tid, task in tarkov.tasks.items():
            name = tarkov.task_name(task)
            parts = []
            for obj in task.get("objectives", []):
                desc = tarkov.tasks_zh.get(obj.get("description", ""), "")
                otype = obj.get("type", "")
                items = obj.get("items", [])
                count = obj.get("count", 1)
                # 涉及物品的目标：解析物品中文名，拼成"需要 X"
                item_names = []
                for iid in items:
                    nm = tarkov.name_by_id(iid) or tarkov.quest_item_name(iid)
                    if nm:
                        item_names.append(nm)
                if desc and item_names:
                    verb = OBJECTIVE_VERB.get(otype, "获取")
                    parts.append(f"{desc}（需要{verb} {'、'.join(item_names)} x{count}）")
                elif desc:
                    parts.append(desc)
            if not parts:
                continue
            text = f"任务「{name}」需求：" + "；".join(parts)
            docs.append({"text": text, "meta": {
                "type": "task", "id": tid, "name": name,
                "normalizedName": task.get("normalizedName", ""),
            }})
        return docs

    def _craft_docs(self):
        """制造配方：每条配方一段"""
        docs = []
        for c in tarkov.crafts:
            station = tarkov.hideout_name(c.get("station", ""))
            level = c.get("level", 1)
            prod = c.get("productItem", {})
            prod_name = tarkov.name_by_id(prod.get("item", "")) or prod.get("item", "")
            prod_count = prod.get("count", 1)
            req_names = []
            for ri in c.get("requiredItems", []):
                nm = tarkov.name_by_id(ri.get("item", "")) or ri.get("item", "")
                req_names.append(f"{nm}x{ri.get('count', 1)}")
            dur = c.get("duration", 0)
            text = (f"藏身处制造配方：在{station}（Lv{level}）制造 {prod_name}x{prod_count}，"
                    f"需要材料 {', '.join(req_names) or '无'}，耗时 {dur} 秒。")
            docs.append({"text": text, "meta": {"type": "craft", "id": c.get("id"), "name": prod_name}})
        return docs

    def _barter_docs(self):
        """以物换物：每条一段"""
        docs = []
        for b in tarkov.barters:
            trader = tarkov.trader_name(b.get("trader", ""))
            offered = b.get("offeredItem", {})
            off_name = tarkov.name_by_id(offered.get("item", "")) or offered.get("item", "")
            req_names = []
            for ri in b.get("requiredItems", []):
                nm = tarkov.name_by_id(ri.get("item", "")) or ri.get("item", "")
                req_names.append(f"{nm}x{ri.get('count', 1)}")
            text = f"商人以物换物：找{trader}，用 {', '.join(req_names) or '货币'} 换取 {off_name}x{offered.get('count', 1)}。"
            docs.append({"text": text, "meta": {"type": "barter", "id": b.get("id"), "name": off_name}})
        return docs

    def _hideout_docs(self):
        """藏身处工作站：每个工作站一段（含各级升级材料）"""
        docs = []
        for sid, st in tarkov.hideout.items():
            name = tarkov.hideout_name(sid)
            level_parts = []
            for lv in st.get("levels", []):
                req_names = []
                for ri in lv.get("itemRequirements", []):
                    nm = tarkov.name_by_id(ri.get("item", "")) or ri.get("item", "")
                    req_names.append(f"{nm}x{ri.get('count', 1)}")
                lv_num = lv.get("level", 1)
                level_parts.append(f"Lv{lv_num}需 {', '.join(req_names) if req_names else '无材料'}")
            text = f"藏身处工作站「{name}」升级需求：" + "；".join(level_parts) + "。"
            docs.append({"text": text, "meta": {"type": "hideout", "id": sid, "name": name}})
        return docs

    # ---------- 检索 ----------
    def _exact_title_hits(self, query: str) -> set:
        """全库扫描：标题（中文名/英文规范名）原样出现在查询里的文档下标。

        这是「用户明确点名了某个物品/任务」的强信号，优先级高于语义相似度。
        纯字符串比对，7000 条文档成本可忽略。
        """
        if not query:
            return set()
        q_lower = query.lower()
        hits = set()
        for i, d in enumerate(self.docs):
            meta = d.get("meta") or {}
            for field in TITLE_FIELDS:
                title = str(meta.get(field, "")).lower()
                if _substring_eligible(title) and title in q_lower:
                    hits.add(i)
                    break
        return hits

    def _expand_query(self, query: str) -> str:
        """把查询里的英文缩写（LEDX/GPU/BTC 等）展开成中文全名，提升检索命中。

        bge 中文模型对纯英文缩写语义理解弱，导致「哪些任务要 LEDX」检索不到
        需要 LEDX 的任务。这里把英文 token 精确匹配成物品中文全名，追加到查询末尾。
        """
        import re
        tokens = re.findall(r'[A-Za-z][A-Za-z0-9]*', query)
        extra = []
        seen = set()
        try:
            for tok in tokens:
                key = tok.lower()
                if key in seen:
                    continue
                seen.add(key)
                cands = tarkov.search(tok, limit=5)
                # 精确命中（英文规范名或短名完全一致）才采用，避免误展开
                exact = [c for c in cands
                         if c.get("normalizedName", "").lower() == key
                         or c.get("shortName", "").lower() == key]
                if exact:
                    nm = exact[0].get("name", "")
                    if nm and nm not in query and nm not in extra:
                        extra.append(nm)
        except Exception:
            pass  # 展开失败不影响检索
        return query + (" " + " ".join(extra) if extra else "")

    def search(self, query: str, top_k: int = 5):
        """语义检索：返回 top_k 个相关文档（含相似度分数）。首次调用自动构建知识库。"""
        if not self._built:
            self.build()
        if self.vectors is None or len(self.docs) == 0:
            return []
        query = self._expand_query(query)  # 英文缩写 → 中文全名，提升命中
        try:
            model = self._get_model()
            qvec = list(model.embed([QUERY_PREFIX + query]))[0].astype(np.float32)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
            # 余弦相似度 = 归一化后点积
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            sims = (self.vectors @ qvec) / (norms[:, 0] + 1e-9)

            # 先多取一批候选，再用词法加权重排。
            # 池子取大一些：语义排序对短查询/英文名不可靠，需要足够候选让词法信号起作用。
            pool = min(len(self.docs), max(top_k * 60, 300))
            cand_idx = set(int(i) for i in np.argsort(-sims)[:pool])

            # 标题直查：全库扫一遍「标题是否原样出现在查询里」。
            # 为什么必须全库扫：像「Collector 任务」，目标文档的语义排名在 1400 名开外，
            # 任何有限候选池都捞不到它，只有标题字面命中才能救回来。
            # 代价极低——纯字符串包含判断，7000 条约几毫秒。
            cand_idx |= self._exact_title_hits(query)

            ranked = []
            for i in cand_idx:
                d = self.docs[i]
                semantic = float(sims[i])
                lexical = _title_lexical_score(query, d.get("meta", {}))
                ranked.append((semantic + LEXICAL_WEIGHT * lexical, semantic, d))
            ranked.sort(key=lambda x: -x[0])

            results = []
            for _, semantic, d in ranked[:top_k]:
                results.append({
                    "text": d["text"],
                    "meta": d["meta"],
                    "score": semantic,   # 仍上报语义分，便于观察真实相似度
                })
            return results
        except Exception as e:
            print(f"[知识库] 检索失败：{e}")
            return []

    # ---------- 缓存 ----------
    def _save_cache(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(DOCS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.docs, f, ensure_ascii=False)
            np.save(VECS_FILE, self.vectors)
        except Exception as e:
            print(f"[知识库] 缓存写入失败：{e}")

    def _load_cache(self):
        try:
            if not DOCS_FILE.exists() or not VECS_FILE.exists():
                return False
            with open(DOCS_FILE, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            self.vectors = np.load(VECS_FILE)
            return len(self.docs) > 0 and self.vectors.shape[0] == len(self.docs)
        except Exception:
            return False


# 全局单例，所有模块共用同一份知识库
kb = KnowledgeBase()
