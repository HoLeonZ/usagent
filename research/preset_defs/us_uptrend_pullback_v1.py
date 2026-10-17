"""preset 定义体：us_uptrend_pullback_v1（见 research/preset_registry.py 的装载契约）。
"""
from __future__ import annotations

from core import costs
from research import universe as mod_universe
from research.preset_common import (DEFAULT_RISK_PER_TRADE, RISK_LIMITS,
                                    V21_PAYOFF_GEOMETRY, V21_SIGNAL_PARAMS)

NAME = "us_uptrend_pullback_v1"

PRESET = {
        # ---- 数据口径 ----------------------------------------------------
        # PIT 取池：截至 T 的滚动成交额门槛，因果（见 research/universe.py）。
        # 曾经这里是字符串 "oi100" —— 一份**今日**的期权 OI 快照，而引擎根本不读它。
        "universe": dict(mod_universe.DEFAULT_SPEC),
        "universe_note": "PIT 流动性池(截至 T 的 60 日滚动成交额 >= $3e7, 因果) —— "
                     "仍未消除的是**退市名单缺失**:DB 里没有已退市标的,所以绝对收益应读作上界",

        # ---- 执行语义(CLAUDE.md §2/§4 的可移植部分)----------------------
        "intraday_tiebreak": "sl_first",
        "max_volume_participation": costs.MAX_VOLUME_PARTICIPATION,
        "cost_bps_roundtrip": costs.COST_BPS_ROUNDTRIP,   # 美股无印花税;IBKR+SECTAF
    # 价差是**尺寸无关**的那一半成本，此前完全没建模（审计 E1）。
    # 用 Corwin-Schultz 从日线 high/low 估计（逐日、因果、系统性低估 ⇒ 是下限）。
    "spread_model": "corwin_schultz",
    # 止损穿价系数：0 = 按触发价成交（既有行为）。**为什么不取 0.5**：实测穿价中位
    # 约 1% / p90 约 1.8%（见 broker 的 sl_penetration 读数），θ 从 0 到 1 会让
    # 头条差 1~3.5pp —— 也就是说这条假定值钱。但 θ 的正确值需要**实盘成交**来标定
    # （台账 G7 的前向验证闭环），在没有标定数据之前拍一个 0.5 就是凭空造假设。
    # 所以：默认 0、穿价**逐笔记录并进指标**，读者自己按 θ∈{0,0.5,1} 跑敏感性。
    "sl_overshoot": 0.0,
        # 执行滑点(§4(c) 要求它必须是 f(ATR, 参与率),不得为零、不得为平坦费率):
        #     slip = scale × 信号日 atr_pct × sqrt(该单股数 / 本 bar 成交量)
        # 这是标准平方根冲击律。scale=1.0 取 Almgren 类经验区间 [0.5, 1] 的**保守端**
        # (「吃掉整根 bar 的量」付一个 ATR)。系数是**先在机制上定死再看结果**的,
        # 不是调参;参与率按量能上限截断,故滑点天然有上界。
        # 2026-10-15 实测(本 preset 41 笔):**入场**参与率中位 0.016%(最大 0.14%)、
        # 信号日 atr_pct 中位 7.45% ⇒ 入场滑点中位 0.085%/边;开与不开的 CAGR 差
        # −3.21% ↔ −4.34%。
        # **出场侧同一公式**(卖出方向压低成交价):出场参与率中位 0.013% ⇒ 出场滑点
        # 中位 0.085%/边。不建模出场滑点会让每笔止损恒等于 −1R,而 R 分布退化会使搜索
        # 系统性偏爱窄止损(窄止损的损失被机械封顶,真实成交的相对损耗反而更大)。
        # (初版注释把 12m preset 的参与率 0.1% 挪到了这里,复核实测已改正。)
        # 此前旧公式把参与率写成字面量 1.0,等于每笔固定付一个 ATR,才会出现
        # 「一根旋钮把 CAGR 从 +44.5% 打到 −28.9%」。
        "atr_slip_scale": costs.ATR_SLIP_SCALE,

        # ---- 仓位政策(偏离 hiagent §2 的 all-in,理由见下)----------------
        "position_sizing": "risk_budget",
        # §2(a) 要求显式声明。注意它是**派生量**:真正决定政策的是 risk_per_trade
        # 这个键在不在(见 core.preset_contract.derived_position_sizing)。
        # 这里声明 1.0 的含义是「若关掉风险预算,就按 §2(c) 的 all-in 走」。
        "position_fraction": 1.0,
        "risk_per_trade": DEFAULT_RISK_PER_TRADE,
        "max_positions": 1,
        **RISK_LIMITS,                     # 组合层风险政策（见上方 RISK_LIMITS）
        # ⚠️ 偏离 hiagent §2「All-In Sizing Policy」。该政策与「窄止损」耦合:
        #    §2 原话 "Tail risk is managed entirely by exit logic (TP/SL), never by
        #    reserving cash at entry" 只在止损很窄时成立。A 股 2.1% 止损下 all-in
        #    每次亏 2.1% 权益;止损加宽到 1 ATR(美股中位 6.79%)后每次亏 6.79%,
        #    连亏 20 次从 −34.6% 变 −75.5%。风险预算式让单笔亏损与止损宽度脱钩。

        # ---- 退出(ATR 归一化,覆盖固定百分比)----------------------------
        # k=1.0 是**机制选择,不是 CAGR 选择**:1 ATR 是「一天典型波幅」的自然单位,
        # 且是能逃出噪声带的最小值;迭代期 E[R] 在 k∈[1.0,2.0] 上基本持平
        # (+0.197 / +0.190 / +0.146),没有梯度的形状就不该去挑峰。
        "atr_sl_mult": 1.0,
        "atr_tp_mult": 1.0 * V21_PAYOFF_GEOMETRY,
        "max_hold": 18,
        # ⚠️ 已知设计缺陷(未修,因为修它需要再一轮搜索,而边缘本就不存在):
        #    TP = 12.514 × 1 ATR ≈ +85%,在 18 天持有期内几乎不可达。
        #    结果:holdout 22 笔中 21 笔走止损、0 笔走 TP,E[R] ≈ −1 是几何的
        #    必然后果而非市场发现。12.5:1 这个比例同样是 A 股校准量。

        # ---- 信号 --------------------------------------------------------
        "signal": dict(V21_SIGNAL_PARAMS),

        # ---- 大盘择时(compute_regime 默认参数,不调参)--------------------
        "regime": {
            "ma_window": 20,
            "breadth_window": 20,
            "min_breadth": 0.45,
            "require_rising": False,      # preset_search §5 M4 已证伪,不要打开
        },

        # ---- §5 记账 -----------------------------------------------------
        # 2(流动性口径候选) × 3(成本档敏感性) = 6
        # + 6(k 的 ATR 阶梯:0.5/1.0/1.5/2.0/3.0/4.0)
        # + 1(风险预算式仓位这一条政策检验)
        # + 1(H1 财报窗口过滤)
        # + 1(H2 移动止损)
        # = 15;本文件的写作、窗口起止的选定本身也计入,故再 +1。
        # 2026-10-15 审计后,**按同一条规则**补记:
        #   计 1 的只有「存在合法替代取值的口径决策」——本次只有「执行滑点从零改为
        #   f(ATR, 参与率)」这一条(系数取 0.5 还是 1.0 是一个选择)。
        #   不计的是**铁律强制、不存在第二个合法取值**的修正:§2(d) 结算隔离、
        #   §3(b) 双价格体系、§3(c) 分红入账、amount 改用 as-traded 价 —— 它们是把
        #   物理上不可能的记账改对,不属于 §5(d) 所说的 iteration。
        #   故 16 → 17(v1)、106 → 107(12m)。
        "n_comparisons": 17,

        # ---- H1(财报窗口过滤)已检验;⚠️ 证据方向与旧版**相反** -------------
        # 2026-10-15 在当前引擎上重测:
        #   全期    无过滤 41 笔 E[R] −0.254 收益 −10.54%  →  有过滤 30 笔 E[R] +0.250 收益 +7.08%
        #   迭代期  18 → 12 笔,E[R] +0.364 → +1.142(收益 +7.80% → +15.84%)
        #   holdout 23 → 19 笔,E[R] −0.981 → −0.676(仍为负,收益 −20.26% → −12.15%)
        # 机制检验(预注册判据:若财报真通过「跳空穿透止损」造成伤害,被剔除批的 sl_gap
        # 占比必须显著更高):被剔除批 2/15 = 13.3%,保留批 1/30 = 3.3% —— 方向**支持** H1。
        # 但 n 只有 2 与 1,**不足以判定因果**。旧版写的是「6.7% vs 6.9%,几乎相同」,
        # 那条「机制检验推翻了它」的论证在修复后的引擎上已不成立。
        # → 仍**不作为默认开启**,但理由改了:不是「机制被推翻」,而是「holdout 仍为负
        #   且机制证据样本过小 —— 缺乏样本外支持」。要开启必须先做它自己的 walk-forward。
        #   工具保留(signals.filter_earnings_window)。
        "avoid_earnings_forward_calendar_days": 26,
        "earnings_filter_validated": False,

        # ---- H2(移动止损)已检验,不成立(数字已按当前引擎重测)--------------
        # 全期 E[R] −0.254 → +0.016、胜率 14.6% → 37.8%,看似改善;
        # 但**两个期间方向相反**(迭代期 +0.364 → +0.231 变差,holdout −0.981 →
        # −0.360 变好),且成交 41 → 74、平均持有 6.63 → 2.42 天、出场退化成
        # SL 64 / sl_gap 10 / TP 0 / max_hold 0 —— 不是更有效率,是变成了高频 churn。
        # 同名现象见 §1B 的 k 阶梯:改善总与样本膨胀同步出现。
        # → 默认关闭。机制与因果性测试保留(tests/test_broker.py 的 H2 四条)。
        "trailing_sl": False,
        "trailing_sl_validated": False,

        # ---- 有效性声明(不要删)------------------------------------------
        "validated": False,
            }

VERDICT = (
    '无可证实的正 alpha。五组假设(A v21 原样 / B 止损 ATR 归一化 / C 风险预算仓位 / H1 财报窗口 / H2 移动止损)逐一被排除,详见 results/BASELINE.md §3。全期超额 −117.13pp,TP 出场 0 笔(2026-10-15 重测)。不得用于实盘,不得当作选股能力的证据。'
)

LOCK_WINDOW = ("2024-04-10", "2026-10-17")
LOCK_IS_CLI_DEFAULT = True
FIXTURE_WINDOW = ("2024-01-01", "2025-01-01")
