from pathlib import Path

import matplotlib

# 在 VS Code 终端运行时不弹出额外窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# 0. 可修改设置区
# 后续修改字体、颜色、位置时，优先调整这里
# ============================================================

FIGURE_WIDTH = 20
FIGURE_HEIGHT = 10
OUTPUT_DPI = 180

TITLE_SIZE = 24
SUBTITLE_SIZE = 11
SECTION_TITLE_SIZE = 13
BOX_TITLE_SIZE = 11
BOX_TEXT_SIZE = 9
NOTE_SIZE = 9

BOX_EDGE_WIDTH = 1.5
ARROW_WIDTH = 1.8

COLORS = {
    "background": "#F6F8FB",
    "title": "#1F2937",
    "subtitle": "#4B5563",
    "section_border": "#CBD5E1",
    "section_fill": "#FFFFFF",
    "data": "#DBEAFE",
    "processing": "#E0F2FE",
    "model": "#EDE9FE",
    "risk": "#FEF3C7",
    "api": "#DCFCE7",
    "dashboard": "#FCE7F3",
    "docker": "#E2E8F0",
    "arrow": "#475569",
    "note_fill": "#FFF7ED",
    "note_border": "#FDBA74",
}


# ============================================================
# 1. 项目路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs" / "architecture"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "financial_risk_system_architecture.png"
SVG_PATH = OUTPUT_DIR / "financial_risk_system_architecture.svg"


# ============================================================
# 2. 绘图工具函数
# ============================================================

def draw_section(
    ax,
    x,
    y,
    width,
    height,
    title,
):
    """
    绘制每一个大的项目阶段区域。
    """

    section = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=COLORS["section_border"],
        facecolor=COLORS["section_fill"],
        zorder=1,
    )

    ax.add_patch(section)

    ax.text(
        x + width / 2,
        y + height - 0.35,
        title,
        ha="center",
        va="center",
        fontsize=SECTION_TITLE_SIZE,
        fontweight="bold",
        color=COLORS["title"],
        zorder=3,
    )


def draw_box(
    ax,
    x,
    y,
    width,
    height,
    title,
    lines,
    fill_color,
):
    """
    绘制阶段内部的功能框。
    """

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=BOX_EDGE_WIDTH,
        edgecolor=COLORS["section_border"],
        facecolor=fill_color,
        zorder=4,
    )

    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height - 0.28,
        title,
        ha="center",
        va="center",
        fontsize=BOX_TITLE_SIZE,
        fontweight="bold",
        color=COLORS["title"],
        zorder=5,
    )

    body_text = "\n".join(lines)

    ax.text(
        x + width / 2,
        y + height / 2 - 0.18,
        body_text,
        ha="center",
        va="center",
        fontsize=BOX_TEXT_SIZE,
        color=COLORS["subtitle"],
        linespacing=1.45,
        zorder=5,
    )


def draw_arrow(
    ax,
    start,
    end,
    dashed=False,
):
    """
    绘制阶段之间的箭头。
    start / end 格式为：(x, y)
    """

    line_style = "--" if dashed else "-"

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=ARROW_WIDTH,
        linestyle=line_style,
        color=COLORS["arrow"],
        connectionstyle="arc3,rad=0.0",
        shrinkA=2,
        shrinkB=2,
        zorder=7,
    )

    ax.add_patch(arrow)


# ============================================================
# 3. 创建画布
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
)

fig.patch.set_facecolor(COLORS["background"])
ax.set_facecolor(COLORS["background"])

ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis("off")


# ============================================================
# 4. 标题与副标题
# ============================================================

ax.text(
    10,
    9.55,
    "End-to-End Financial Transaction Risk Monitoring System",
    ha="center",
    va="center",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    color=COLORS["title"],
)

ax.text(
    10,
    9.08,
    (
        "Synthetic transaction data | SQL analytics | Machine learning | "
        "Risk scoring | FastAPI | Streamlit | Docker Compose"
    ),
    ha="center",
    va="center",
    fontsize=SUBTITLE_SIZE,
    color=COLORS["subtitle"],
)


# ============================================================
# 5. 大阶段区域
# ============================================================

SECTION_Y = 2.0
SECTION_HEIGHT = 6.45

sections = {
    "data": {
        "x": 0.4,
        "w": 2.55,
        "title": "1. Data Source",
    },
    "processing": {
        "x": 3.15,
        "w": 3.05,
        "title": "2. Data Processing",
    },
    "model": {
        "x": 6.4,
        "w": 3.35,
        "title": "3. Modeling",
    },
    "risk": {
        "x": 9.95,
        "w": 2.9,
        "title": "4. Risk Decision",
    },
    "api": {
        "x": 13.05,
        "w": 2.75,
        "title": "5. Model Serving",
    },
    "deployment": {
        "x": 16.0,
        "w": 3.6,
        "title": "6. Monitoring & Deployment",
    },
}

for section in sections.values():
    draw_section(
        ax=ax,
        x=section["x"],
        y=SECTION_Y,
        width=section["w"],
        height=SECTION_HEIGHT,
        title=section["title"],
    )


# ============================================================
# 6. Data Source
# ============================================================

draw_box(
    ax,
    x=0.72,
    y=5.3,
    width=1.9,
    height=2.15,
    title="Synthetic Dataset",
    lines=[
        "50,000 transactions",
        "21 raw variables",
        "2023 transaction period",
        "Fraud label available",
    ],
    fill_color=COLORS["data"],
)

draw_box(
    ax,
    x=0.72,
    y=2.75,
    width=1.9,
    height=1.75,
    title="Raw Variables",
    lines=[
        "Amount and balance",
        "Device and location",
        "Transaction behaviour",
        "Authentication signals",
    ],
    fill_color=COLORS["data"],
)


# ============================================================
# 7. Data Processing
# ============================================================

draw_box(
    ax,
    x=3.48,
    y=6.05,
    width=2.38,
    height=1.45,
    title="Cleaning & Validation",
    lines=[
        "Column standardisation",
        "Missing / duplicate checks",
        "Timestamp conversion",
    ],
    fill_color=COLORS["processing"],
)

draw_box(
    ax,
    x=3.48,
    y=4.3,
    width=2.38,
    height=1.35,
    title="Feature Engineering",
    lines=[
        "Year, month and hour",
        "Day of week and weekend",
        "Behavioural variables",
    ],
    fill_color=COLORS["processing"],
)

draw_box(
    ax,
    x=3.48,
    y=2.55,
    width=2.38,
    height=1.35,
    title="SQL & EDA",
    lines=[
        "SQLite transaction table",
        "User risk aggregation",
        "Fraud pattern analysis",
    ],
    fill_color=COLORS["processing"],
)


# ============================================================
# 8. Modeling
# ============================================================

draw_box(
    ax,
    x=6.73,
    y=5.55,
    width=2.69,
    height=1.95,
    title="Supervised Models",
    lines=[
        "Logistic Regression",
        "Random Forest",
        "Balanced Random Forest",
        "Weighted XGBoost",
    ],
    fill_color=COLORS["model"],
)

draw_box(
    ax,
    x=6.73,
    y=3.7,
    width=2.69,
    height=1.4,
    title="Model Evaluation",
    lines=[
        "Precision and recall",
        "ROC-AUC and PR-AUC",
        "Confusion matrix",
    ],
    fill_color=COLORS["model"],
)

draw_box(
    ax,
    x=6.73,
    y=2.45,
    width=2.69,
    height=0.9,
    title="Anomaly Detection",
    lines=[
        "Isolation Forest",
    ],
    fill_color=COLORS["model"],
)


# ============================================================
# 9. Risk Decision
# ============================================================

draw_box(
    ax,
    x=10.28,
    y=5.8,
    width=2.24,
    height=1.7,
    title="Threshold Analysis",
    lines=[
        "Precision-recall trade-off",
        "Review workload",
        "Missed fraud analysis",
    ],
    fill_color=COLORS["risk"],
)

draw_box(
    ax,
    x=10.28,
    y=3.75,
    width=2.24,
    height=1.65,
    title="Risk Score",
    lines=[
        "Probability × 100",
        "Low: below 20",
        "Medium: 20 to 70",
        "High: 70 or above",
    ],
    fill_color=COLORS["risk"],
)

draw_box(
    ax,
    x=10.28,
    y=2.45,
    width=2.24,
    height=0.9,
    title="Operational Action",
    lines=[
        "Approve / Monitor / Review",
    ],
    fill_color=COLORS["risk"],
)


# ============================================================
# 10. Model Serving
# ============================================================

draw_box(
    ax,
    x=13.38,
    y=5.65,
    width=2.09,
    height=1.85,
    title="Saved Pipeline",
    lines=[
        "Preprocessing",
        "Feature encoding",
        "Random Forest model",
        "Joblib serialisation",
    ],
    fill_color=COLORS["api"],
)

draw_box(
    ax,
    x=13.38,
    y=3.65,
    width=2.09,
    height=1.55,
    title="FastAPI Service",
    lines=[
        "GET /health",
        "POST /predict",
        "Pydantic validation",
    ],
    fill_color=COLORS["api"],
)

draw_box(
    ax,
    x=13.38,
    y=2.45,
    width=2.09,
    height=0.8,
    title="JSON Response",
    lines=[
        "Probability + action",
    ],
    fill_color=COLORS["api"],
)


# ============================================================
# 11. Monitoring and Deployment
# ============================================================

draw_box(
    ax,
    x=16.35,
    y=5.45,
    width=2.9,
    height=2.05,
    title="Streamlit Dashboard",
    lines=[
        "Single-transaction scoring",
        "Model performance",
        "Risk monitoring overview",
        "High-risk transaction list",
    ],
    fill_color=COLORS["dashboard"],
)

draw_box(
    ax,
    x=16.35,
    y=3.6,
    width=2.9,
    height=1.4,
    title="Docker Compose",
    lines=[
        "FastAPI container",
        "Streamlit container",
        "Internal service network",
    ],
    fill_color=COLORS["docker"],
)

draw_box(
    ax,
    x=16.35,
    y=2.45,
    width=2.9,
    height=0.75,
    title="Health Check",
    lines=[
        "API starts before dashboard",
    ],
    fill_color=COLORS["docker"],
)


# ============================================================
# 12. 主流程箭头
# ============================================================

draw_arrow(
    ax,
    start=(2.63, 5.3),
    end=(3.45, 5.3),
)

draw_arrow(
    ax,
    start=(5.88, 5.3),
    end=(6.7, 5.3),
)

draw_arrow(
    ax,
    start=(9.44, 5.3),
    end=(10.25, 5.3),
)

draw_arrow(
    ax,
    start=(12.54, 5.3),
    end=(13.35, 5.3),
)

draw_arrow(
    ax,
    start=(15.49, 5.3),
    end=(16.32, 5.3),
)


# ============================================================
# 13. 阶段内部箭头
# ============================================================

draw_arrow(
    ax,
    start=(4.67, 6.03),
    end=(4.67, 5.67),
)

draw_arrow(
    ax,
    start=(4.67, 4.28),
    end=(4.67, 3.92),
)

draw_arrow(
    ax,
    start=(8.08, 5.53),
    end=(8.08, 5.12),
)

draw_arrow(
    ax,
    start=(11.4, 5.78),
    end=(11.4, 5.42),
)

draw_arrow(
    ax,
    start=(11.4, 3.72),
    end=(11.4, 3.37),
)

draw_arrow(
    ax,
    start=(14.42, 5.62),
    end=(14.42, 5.22),
)

draw_arrow(
    ax,
    start=(14.42, 3.62),
    end=(14.42, 3.28),
)

draw_arrow(
    ax,
    start=(17.8, 5.42),
    end=(17.8, 5.02),
)

draw_arrow(
    ax,
    start=(17.8, 3.57),
    end=(17.8, 3.22),
)


# ============================================================
# 14. 底部说明
# ============================================================

note = FancyBboxPatch(
    (0.65, 0.65),
    18.7,
    0.82,
    boxstyle="round,pad=0.03,rounding_size=0.08",
    linewidth=1.2,
    edgecolor=COLORS["note_border"],
    facecolor=COLORS["note_fill"],
    zorder=2,
)

ax.add_patch(note)

ax.text(
    10,
    1.06,
    (
        "Proof-of-concept using synthetic data. "
        "Held-out test metrics demonstrate the workflow and are not production guarantees."
    ),
    ha="center",
    va="center",
    fontsize=NOTE_SIZE,
    color=COLORS["subtitle"],
    zorder=3,
)


# ============================================================
# 15. 保存图片
# ============================================================

plt.savefig(
    PNG_PATH,
    dpi=OUTPUT_DPI,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.savefig(
    SVG_PATH,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

plt.close(fig)

print("Architecture diagram generated successfully.")
print(f"PNG: {PNG_PATH}")
print(f"SVG: {SVG_PATH}")