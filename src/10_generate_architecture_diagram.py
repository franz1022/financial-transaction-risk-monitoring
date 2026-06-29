from pathlib import Path

import matplotlib

# VSCode / PowerShell 终端运行时不弹出窗口
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# 0. 可修改设置区
# ============================================================

FIGURE_WIDTH = 22
FIGURE_HEIGHT = 11
OUTPUT_DPI = 180

TITLE_SIZE = 23
SUBTITLE_SIZE = 11
SECTION_TITLE_SIZE = 12
BOX_TITLE_SIZE = 10.5
BOX_TEXT_SIZE = 8.7
NOTE_SIZE = 9

BOX_EDGE_WIDTH = 1.4
ARROW_WIDTH = 1.8

COLORS = {
    "background": "#F6F8FB",
    "title": "#1F2937",
    "subtitle": "#4B5563",
    "section_border": "#CBD5E1",
    "section_fill": "#FFFFFF",
    "data": "#DBEAFE",
    "audit": "#FDE68A",
    "validation": "#EDE9FE",
    "policy": "#FCE7F3",
    "api": "#DCFCE7",
    "deployment": "#E2E8F0",
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
# 2. 绘图函数
# ============================================================

def draw_section(ax, x, y, width, height, title):
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
        y + height - 0.34,
        title,
        ha="center",
        va="center",
        fontsize=SECTION_TITLE_SIZE,
        fontweight="bold",
        color=COLORS["title"],
        zorder=3,
    )


def draw_box(ax, x, y, width, height, title, lines, fill_color):
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
        y + height - 0.27,
        title,
        ha="center",
        va="center",
        fontsize=BOX_TITLE_SIZE,
        fontweight="bold",
        color=COLORS["title"],
        zorder=5,
    )

    ax.text(
        x + width / 2,
        y + height / 2 - 0.16,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=BOX_TEXT_SIZE,
        color=COLORS["subtitle"],
        linespacing=1.42,
        zorder=5,
    )


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=ARROW_WIDTH,
        color=COLORS["arrow"],
        connectionstyle="arc3,rad=0.0",
        shrinkA=2,
        shrinkB=2,
        zorder=7,
    )
    ax.add_patch(arrow)


# ============================================================
# 3. 画布
# ============================================================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
fig.patch.set_facecolor(COLORS["background"])
ax.set_facecolor(COLORS["background"])
ax.set_xlim(0, 22)
ax.set_ylim(0, 11)
ax.axis("off")


# ============================================================
# 4. 标题
# ============================================================

ax.text(
    11,
    10.5,
    "Governance-Aware Financial Transaction Risk Monitoring System",
    ha="center",
    va="center",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    color=COLORS["title"],
)

ax.text(
    11,
    10.03,
    (
        "Synthetic data | Proxy-feature audit | Temporal validation | "
        "Diagnostic-only model | FastAPI | Streamlit | Docker Compose"
    ),
    ha="center",
    va="center",
    fontsize=SUBTITLE_SIZE,
    color=COLORS["subtitle"],
)


# ============================================================
# 5. 六个阶段
# ============================================================

SECTION_Y = 2.05
SECTION_HEIGHT = 7.25

sections = [
    (0.35, 3.05, "1. Data & Analytics"),
    (3.60, 3.25, "2. Model-Risk Audit"),
    (7.05, 3.55, "3. Temporal Validation"),
    (10.80, 3.45, "4. Decision Policy"),
    (14.45, 3.10, "5. Governance API"),
    (17.75, 3.90, "6. Monitoring & Deployment"),
]

for x, width, title in sections:
    draw_section(
        ax,
        x,
        SECTION_Y,
        width,
        SECTION_HEIGHT,
        title,
    )


# ============================================================
# 6. Data & Analytics
# ============================================================

draw_box(
    ax, 0.70, 6.45, 2.35, 1.85,
    "Synthetic Dataset",
    [
        "50,000 transactions",
        "21 original variables",
        "2023 transaction period",
        "Fraud label available",
    ],
    COLORS["data"],
)

draw_box(
    ax, 0.70, 4.45, 2.35, 1.55,
    "Cleaning & Features",
    [
        "Missing / duplicate checks",
        "Timestamp features",
        "Behavioural variables",
    ],
    COLORS["data"],
)

draw_box(
    ax, 0.70, 2.65, 2.35, 1.35,
    "SQL & EDA",
    [
        "Fraud pattern analysis",
        "User-level aggregation",
        "Rule-based summaries",
    ],
    COLORS["data"],
)


# ============================================================
# 7. Model-Risk Audit
# ============================================================

draw_box(
    ax, 3.95, 6.35, 2.55, 1.95,
    "Feature Leakage Audit",
    [
        "risk_score excluded",
        "Proxy-feature checks",
        "User overlap analysis",
        "Split integrity review",
    ],
    COLORS["audit"],
)

draw_box(
    ax, 3.95, 4.35, 2.55, 1.55,
    "Deterministic Proxy",
    [
        "failed_count >= 4",
        "100% synthetic fraud rate",
        "Not real-world evidence",
    ],
    COLORS["audit"],
)

draw_box(
    ax, 3.95, 2.65, 2.55, 1.25,
    "Audit Outcome",
    [
        "Inflated benchmark detected",
        "Deployment claim rejected",
    ],
    COLORS["audit"],
)


# ============================================================
# 8. Temporal Validation
# ============================================================

draw_box(
    ax, 7.42, 6.45, 2.82, 1.85,
    "Chronological Split",
    [
        "Train: earliest 60%",
        "Validation: next 20%",
        "Test: latest 20%",
        "No random split",
    ],
    COLORS["validation"],
)

draw_box(
    ax, 7.42, 4.25, 2.82, 1.75,
    "Feature Ablation",
    [
        "Benchmark feature set",
        "Conservative feature set",
        "Proxy contribution isolated",
    ],
    COLORS["validation"],
)

draw_box(
    ax, 7.42, 2.55, 2.82, 1.25,
    "Residual Test",
    [
        "Test ROC-AUC: 0.4875",
        "Test PR-AUC: 0.1458",
    ],
    COLORS["validation"],
)


# ============================================================
# 9. Decision Policy
# ============================================================

draw_box(
    ax, 11.15, 6.25, 2.75, 2.05,
    "Transparent Synthetic Rule",
    [
        "failed_count >= 4",
        "Synthetic Rule Alert",
        "Manual review demo only",
        "Production eligible: No",
    ],
    COLORS["policy"],
)

draw_box(
    ax, 11.15, 4.15, 2.75, 1.65,
    "Residual Diagnostic Model",
    [
        "Logistic pipeline",
        "19 diagnostic features",
        "No decision authority",
    ],
    COLORS["policy"],
)

draw_box(
    ax, 11.15, 2.55, 2.75, 1.15,
    "Governance Decision",
    [
        "Do not deploy",
        "No approve / reject / block",
    ],
    COLORS["policy"],
)


# ============================================================
# 10. Governance API
# ============================================================

draw_box(
    ax, 14.80, 6.35, 2.40, 1.95,
    "Diagnostic Artifact",
    [
        "Preprocessing pipeline",
        "Residual Logistic model",
        "Metadata + policy JSON",
        "Joblib serialisation",
    ],
    COLORS["api"],
)

draw_box(
    ax, 14.80, 4.20, 2.40, 1.70,
    "FastAPI Service",
    [
        "GET /",
        "GET /health",
        "POST /predict",
        "Pydantic validation",
    ],
    COLORS["api"],
)

draw_box(
    ax, 14.80, 2.55, 2.40, 1.15,
    "Governance Response",
    [
        "Rule status + diagnostic score",
        "Deployment eligibility: No",
    ],
    COLORS["api"],
)


# ============================================================
# 11. Monitoring & Deployment
# ============================================================

draw_box(
    ax, 18.15, 6.25, 3.10, 2.05,
    "Streamlit Dashboard",
    [
        "Transaction governance check",
        "Temporal validation evidence",
        "Rule and residual analysis",
        "Governance summary",
    ],
    COLORS["deployment"],
)

draw_box(
    ax, 18.15, 4.25, 3.10, 1.55,
    "Docker Compose",
    [
        "FastAPI container",
        "Streamlit container",
        "API health dependency",
    ],
    COLORS["deployment"],
)

draw_box(
    ax, 18.15, 2.55, 3.10, 1.20,
    "Automated Tests",
    [
        "17 API and validation tests",
        "Input and governance checks",
    ],
    COLORS["deployment"],
)


# ============================================================
# 12. 箭头
# ============================================================

draw_arrow(ax, (3.08, 5.55), (3.92, 5.55))
draw_arrow(ax, (6.53, 5.55), (7.39, 5.55))
draw_arrow(ax, (10.27, 5.55), (11.12, 5.55))
draw_arrow(ax, (13.93, 5.55), (14.77, 5.55))
draw_arrow(ax, (17.23, 5.55), (18.12, 5.55))

draw_arrow(ax, (1.88, 6.42), (1.88, 6.02))
draw_arrow(ax, (1.88, 4.42), (1.88, 4.02))

draw_arrow(ax, (5.23, 6.32), (5.23, 5.92))
draw_arrow(ax, (5.23, 4.32), (5.23, 3.92))

draw_arrow(ax, (8.83, 6.42), (8.83, 6.02))
draw_arrow(ax, (8.83, 4.22), (8.83, 3.82))

draw_arrow(ax, (12.52, 6.22), (12.52, 5.82))
draw_arrow(ax, (12.52, 4.12), (12.52, 3.72))

draw_arrow(ax, (16.00, 6.32), (16.00, 5.92))
draw_arrow(ax, (16.00, 4.17), (16.00, 3.72))

draw_arrow(ax, (19.70, 6.22), (19.70, 5.82))
draw_arrow(ax, (19.70, 4.22), (19.70, 3.77))


# ============================================================
# 13. 底部治理说明
# ============================================================

note = FancyBboxPatch(
    (0.65, 0.65),
    20.70,
    0.90,
    boxstyle="round,pad=0.03,rounding_size=0.08",
    linewidth=1.2,
    edgecolor=COLORS["note_border"],
    facecolor=COLORS["note_fill"],
    zorder=2,
)
ax.add_patch(note)

ax.text(
    11,
    1.10,
    (
        "Portfolio proof of concept using synthetic data. "
        "The deterministic rule and diagnostic model are not approved "
        "for production or customer-impact decisions."
    ),
    ha="center",
    va="center",
    fontsize=NOTE_SIZE,
    color=COLORS["subtitle"],
    zorder=3,
)


# ============================================================
# 14. 保存
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

print("Governance architecture diagram generated successfully.")
print(f"PNG: {PNG_PATH}")
print(f"SVG: {SVG_PATH}")
