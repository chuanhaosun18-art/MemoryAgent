"""
tools.py - 工具模块
====================

提供记忆 Agent 的所有外部工具：

1. web_search() - 使用 SerpAPI 进行 Google 搜索
2. search_restaurants() - 按条件筛选模拟餐厅
3. get_restaurant_detail() - 获取餐厅详情
4. TOOLS - OpenAI Function Calling 格式的工具定义
5. TOOL_FUNCTIONS - 工具名 → 函数的映射表
"""

import json

from serpapi import GoogleSearch
from config import SERPAPI_KEY, SEARCH_MAX_RESULTS


# ============================================
# 网络搜索工具（SerpAPI Google 搜索）
# ============================================

def web_search(query, max_results=SEARCH_MAX_RESULTS):
    """
    使用 SerpAPI (Google) 搜索互联网并返回格式化结果。

    参数:
        query: 搜索关键词
        max_results: 返回结果条数

    返回:
        str: 格式化的搜索结果文本
    """
    try:
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": max_results,
            "hl": "zh-CN",
        })
        data = search.get_dict()
    except Exception as e:
        return json.dumps({"error": f"搜索出错: {e}"}, ensure_ascii=False)

    results = data.get("organic_results", [])
    if not results:
        return json.dumps({"message": "未找到相关搜索结果。"}, ensure_ascii=False)

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "无标题")
        snippet = r.get("snippet", "无摘要")
        link = r.get("link", "")
        formatted.append({
            "index": i,
            "title": title,
            "snippet": snippet,
            "link": link,
        })

    return json.dumps(formatted, ensure_ascii=False)

# ============================================
# 模拟餐厅数据库
# ============================================
# 每家餐厅的字段说明：
#   name: 餐厅名称
#   cuisine: 菜系
#   spicy_level: 辣度 0-3（0=不辣, 1=微辣, 2=中辣, 3=重辣）
#   price: 人均价格（元）
#   rating: 评分（满分5.0）
#   area: 所在区域
#   features: 特色标签
#   has_seafood: 是否含海鲜（用于过敏筛选）
#   vegetarian_friendly: 是否有素食选项
RESTAURANT_DB = [
    {
        "name": "海底捞火锅",
        "cuisine": "火锅",
        "spicy_level": 3,
        "price": 120,
        "rating": 4.5,
        "area": "全国连锁",
        "features": ["服务好", "锅底选择多", "有清汤锅底", "等位有零食"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "鼎泰丰",
        "cuisine": "台菜/点心",
        "spicy_level": 0,
        "price": 150,
        "rating": 4.7,
        "area": "全国连锁",
        "features": ["小笼包", "精致点心", "环境优雅"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "外婆家",
        "cuisine": "杭帮菜",
        "spicy_level": 0,
        "price": 70,
        "rating": 4.3,
        "area": "全国连锁",
        "features": ["性价比高", "家常菜", "茶香鸡"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "西贝莜面村",
        "cuisine": "西北菜",
        "spicy_level": 1,
        "price": 90,
        "rating": 4.4,
        "area": "全国连锁",
        "features": ["莜面", "羊肉", "儿童友好"],
        "has_seafood": False,
        "vegetarian_friendly": True,
    },
    {
        "name": "太二酸菜鱼",
        "cuisine": "川菜",
        "spicy_level": 2,
        "price": 80,
        "rating": 4.2,
        "area": "全国连锁",
        "features": ["酸菜鱼", "只做四人以下桌", "年轻时尚"],
        "has_seafood": True,
        "vegetarian_friendly": False,
    },
    {
        "name": "大董烤鸭店",
        "cuisine": "北京菜",
        "spicy_level": 0,
        "price": 300,
        "rating": 4.6,
        "area": "北京",
        "features": ["烤鸭", "高端", "商务宴请", "意境菜"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "赤坂亭",
        "cuisine": "日料",
        "spicy_level": 0,
        "price": 200,
        "rating": 4.5,
        "area": "全国连锁",
        "features": ["和牛", "铁板烧", "日式烤肉"],
        "has_seafood": True,
        "vegetarian_friendly": False,
    },
    {
        "name": "绿茶餐厅",
        "cuisine": "杭帮菜",
        "spicy_level": 0,
        "price": 65,
        "rating": 4.1,
        "area": "全国连锁",
        "features": ["面包诱惑", "性价比高", "排队多"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "陈记烧鹅",
        "cuisine": "粤菜",
        "spicy_level": 0,
        "price": 85,
        "rating": 4.4,
        "area": "广州",
        "features": ["烧鹅", "老字号", "地道广式"],
        "has_seafood": False,
        "vegetarian_friendly": False,
    },
    {
        "name": "小龙坎火锅",
        "cuisine": "火锅",
        "spicy_level": 3,
        "price": 100,
        "rating": 4.3,
        "area": "全国连锁",
        "features": ["正宗四川火锅", "毛肚", "牛油锅底"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "南京大牌档",
        "cuisine": "淮扬菜",
        "spicy_level": 0,
        "price": 75,
        "rating": 4.3,
        "area": "全国连锁",
        "features": ["盐水鸭", "民国风情", "南京特色"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "木屋烧烤",
        "cuisine": "烧烤",
        "spicy_level": 2,
        "price": 80,
        "rating": 4.0,
        "area": "全国连锁",
        "features": ["烤串", "啤酒", "朋友聚餐"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "萨莉亚",
        "cuisine": "意大利菜",
        "spicy_level": 0,
        "price": 40,
        "rating": 3.9,
        "area": "全国连锁",
        "features": ["便宜", "意面", "披萨", "学生最爱"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "必胜客",
        "cuisine": "西餐",
        "spicy_level": 0,
        "price": 70,
        "rating": 3.8,
        "area": "全国连锁",
        "features": ["披萨", "牛排", "家庭聚餐"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
    {
        "name": "云海肴",
        "cuisine": "云南菜",
        "spicy_level": 1,
        "price": 90,
        "rating": 4.2,
        "area": "全国连锁",
        "features": ["汽锅鸡", "鲜花饼", "民族风情"],
        "has_seafood": False,
        "vegetarian_friendly": True,
    },
    {
        "name": "松鹤楼",
        "cuisine": "苏帮菜",
        "spicy_level": 0,
        "price": 160,
        "rating": 4.5,
        "area": "苏州",
        "features": ["松鼠桂鱼", "百年老店", "正宗苏帮"],
        "has_seafood": True,
        "vegetarian_friendly": True,
    },
]


# ============================================
# 工具函数
# ============================================

def search_restaurants(cuisine=None, area=None, max_price=None):
    """
    搜索餐厅（模拟）。

    根据菜系、地区、最高价格筛选，返回匹配的餐厅列表。
    如果不传任何参数，返回所有餐厅。

    参数:
        cuisine: 菜系名称，如 "川菜"、"日料"
        area: 地区，如 "北京"、"全国连锁"
        max_price: 人均最高价格（元）

    返回:
        str: JSON 格式的餐厅列表
    """
    results = RESTAURANT_DB.copy()

    if cuisine:
        results = [r for r in results if cuisine in r["cuisine"]]

    if area:
        results = [r for r in results
                   if area in r["area"] or r["area"] == "全国连锁"]

    if max_price is not None:
        results = [r for r in results if r["price"] <= max_price]

    # 返回简化信息（不暴露全部字段给 LLM，避免 token 浪费）
    output = []
    for r in results:
        spicy_map = {0: "不辣", 1: "微辣", 2: "中辣", 3: "重辣"}
        output.append({
            "name": r["name"],
            "cuisine": r["cuisine"],
            "spicy_level": spicy_map[r["spicy_level"]],
            "price": f"人均{r['price']}元",
            "rating": r["rating"],
            "area": r["area"],
            "features": r["features"],
        })

    if not output:
        return json.dumps({"message": "没有找到符合条件的餐厅，建议放宽搜索条件"}, ensure_ascii=False)

    return json.dumps(output, ensure_ascii=False)


def get_restaurant_detail(name):
    """
    获取餐厅详情（模拟）。

    参数:
        name: 餐厅名称

    返回:
        str: JSON 格式的餐厅详细信息
    """
    for r in RESTAURANT_DB:
        if r["name"] == name or name in r["name"]:
            spicy_map = {0: "不辣", 1: "微辣", 2: "中辣", 3: "重辣"}
            detail = {
                "name": r["name"],
                "cuisine": r["cuisine"],
                "spicy_level": spicy_map[r["spicy_level"]],
                "price": f"人均{r['price']}元",
                "rating": r["rating"],
                "area": r["area"],
                "features": r["features"],
                "has_seafood": "含海鲜" if r["has_seafood"] else "不含海鲜",
                "vegetarian_friendly": "有素食选项" if r["vegetarian_friendly"] else "无素食选项",
            }
            return json.dumps(detail, ensure_ascii=False)

    return json.dumps({"error": f"未找到名为 '{name}' 的餐厅"}, ensure_ascii=False)


# ============================================
# OpenAI Function Calling 格式的工具定义
# ============================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取实时信息。当用户问到你不确定的事实、最新资讯、天气、新闻、技术问题等需要查找的内容时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，要精准具体，如 '北京今天天气'、'Python 3.12 新特性'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "搜索餐厅。可以按菜系、地区、最高人均价格筛选。不传参数则返回所有餐厅。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "菜系，如 '川菜'、'日料'、'粤菜'、'火锅'、'西餐'",
                    },
                    "area": {
                        "type": "string",
                        "description": "地区，如 '北京'、'上海'、'广州'",
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "人均最高价格（元）",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_detail",
            "description": "获取指定餐厅的详细信息，包括辣度、价格、是否含海鲜、是否有素食等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "餐厅名称，如 '海底捞火锅'、'鼎泰丰'",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

# 工具名 → 函数的映射表
TOOL_FUNCTIONS = {
    "web_search": web_search,
    "search_restaurants": search_restaurants,
    "get_restaurant_detail": get_restaurant_detail,
}
