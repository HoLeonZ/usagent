export const meta = {
  name: 'design-price-action-families',
  description: '为「从价格行为挖低估机会」设计多个互不相同的策略族,并筛选出可实现、可证伪的一组',
  phases: [
    { title: 'Propose', detail: '10 个不同视角各提一个策略族' },
    { title: 'Judge', detail: '3 个评委按机制/重叠度/可实施性打分' },
    { title: 'Synthesize', detail: '合成最终规格(6~8 个互补的族)' },
  ],
}

const CTX = [
  '你在一个**美股量化研究项目**里工作(仓库 /Users/holeon/code/usagent)。',
  '',
  '【任务背景】现有的策略族是「上升趋势中买回调」(uptrend-pullback),它的规则是:',
  '  A 趋势: close > MA60 ∧ 过去60日 close>MA60 的占比 ≥ 0.6693 ∧ mom120 ≥ 0.18',
  '  B 连阴: down_streak ∈ [3,10]',
  '  C 当日跌幅: ret1 ∈ [-8.8%, -2%]',
  '  D MACD 动能衰减: dif>0 ∧ dea>0 ∧ |bar| < |bar_prev|',
  '  E 流动性: amount60 ∈ [3e7, 3e8]',
  '  F 大盘门: 等权指数 > 自身MA20 ∧ 市场宽度 ≥ 0.45',
  '  执行: 信号日 T 收盘 → T+1 开盘买入; ATR 归一化止损/止盈; max_hold 上限;',
  '        每笔计划风险 = 权益 × risk_per_trade(1%); 最多同时持 2 个仓位。',
  '',
  '实测结论(三轮审计 + 一轮预注册 WFV 搜索,全部可复现):',
  '  * 该族在美股这个池子上**没有可证实的绝对收益**:搜索期池化测试窗 CAGR +14.19%、',
  '    连续净值 +8.15%;唯一未参与选择的 holdout(最近 12 个月)上是 -2.19%,跑输基准 18.79pp。',
  '  * DSR(k=138) = 0.021 —— 按选择偏差折算后与噪声不可区分;随机入场对照四窗百分位',
  '    100 / 45.5 / 76.5 / 58(三窗落在随机分布内)。',
  '  * 它的基线(v21 原样)**全期超额 -117pp**。',
  '',
  '【用户的指令】',
  '  「我不想看基本面数据,我只想从**价格行为**中挖掘**低估**的机会并买入。',
  '    如果 uptrend pullback 不行,挖掘其它策略思想,直到能战胜 uptrend pullback 当前的最佳结果为止。」',
  '',
  '【数据与硬约束】',
  '  * 只有**3 年**美股日线(OHLCV),596 只(今日 OI≥100 快照,含幸存者偏差)。可评估 2.51 年。',
  '  * **不许用基本面/财报数字**(用户明确排除)。允许用:价格、成交量、分红(dividends 列)。',
  '  * 数据列算好的有(下游可直接用,全部按 symbol 分组、只用 ≤T 的数据):',
  '    open/high/low/close/volume/amount/dividends、ma5/10/20/60/120、prev_close、ret1、',
  '    above_ma20、above_ma60、above_ma60_ratio、down_streak、prev_down_streak、high20、',
  '    pullback(= close/high20-1)、mom60、mom120、amount60、vol_ma20、vol_ratio、tr、',
  '    atr14、atr_pct、ema12/26、macd_dif/dea/bar/bar_prev、rsi7/rsi14、',
  '    bb_mid/sd/up/lo/pctb/width、stoch_k/stoch_d、plus_di/minus_di/adx14、',
  '    obv/obv_slope/vol_z、hh252/ll252/dist_high252/dist_low252、rvol20、',
  '    xs_mom120/xs_mom60/xs_rvol20(横截面百分位,同日全池排名)。',
  '  * 允许为一个族**新增最多 2 个**因果指标(必须给精确公式,且只能用 ≤T 的数据)。',
  '  * 不能用 df.shift(-x) / bfill / 全表归一化 / 未来函数。',
].join('\n')

const RULES = [
  '【输出纪律】',
  '1. 只提**一个**策略族(你被指派的角度见下),不要罗列一堆。',
  '2. 规则必须**精确到可以照着写代码**:用上面列出的列名表达,给具体阈值。',
  '3. 必须写清**它凭什么与 uptrend-pullback 不同**(机制层面,不是参数层面)——',
  '   如果它只是「换个阈值」或「多加一个条件」,那不是新的策略思想。',
  '4. 必须写清**「低估」在你这个族里是什么意思**:价格行为的什么形态代表「相对便宜」。',
  '5. 必须写清**什么结果会证伪它**(预注册的证伪判据)。',
  '6. 要诚实:如果你认为自己的族大概率无效,明说,并说清为什么还值得一试。',
].join('\n')

const ANGLES = [
  { key: 'absolute-oversold', ask: '角度:**绝对超卖**(个股自身价格处于极端低位,不看大盘不看趋势)' },
  { key: 'cross-sectional', ask: '角度:**横截面**(在同一时点全池里,相对最便宜的那一批;用 xs_* 或排名)' },
  { key: 'volume', ask: '角度:**成交量**(用量的形态判断卖方是否衰竭/买方是否吸筹)' },
  { key: 'volatility', ask: '角度:**波动率**(波动结构本身携带「便宜/恐慌」的信息)' },
  { key: 'candle-shape', ask: '角度:**K线形态**(单根或两三根 bar 的形状本身,如长下影、吞没、十字星)' },
  { key: 'gap', ask: '角度:**跳空**(开盘跳空是最纯粹的「隔夜错价」形态)' },
  { key: 'divergence', ask: '角度:**量价/指标背离**(价格新低但某个量或指标不新低)' },
  { key: 'long-horizon', ask: '角度:**长周期反转**(用 120~252 日的视野,而非 3~10 天)' },
  { key: 'regime-conditional', ask: '角度:**状态依赖**(同一个价格形态在不同市场状态下含义相反;把状态做成条件)' },
  { key: 'structure', ask: '角度:**价格结构**(支撑/前高前低/箱体边界这类「结构位」上的反应,而非均线或指标)' },
]

const SPEC = {
  type: 'object',
  properties: {
    family_id: { type: 'string', description: '英文短名,kebab-case' },
    name: { type: 'string', description: '中文一句话名字' },
    angle: { type: 'string' },
    hypothesis: { type: 'string', description: '机制假设:为什么这个价格行为意味着被低估' },
    rule: { type: 'string', description: '精确规则(用给定列名 + 具体阈值),可直接照着写代码' },
    new_indicator: { type: 'string', description: '需要的新指标(≤2 个,给精确因果公式);没有写 无' },
    grid: { type: 'string', description: '预注册参数网格(每个参数给 3~4 个取值)' },
    holding: { type: 'string', description: '预期持有期与退出几何的直觉' },
    distinct_from_uptrend_pullback: { type: 'string', description: '机制层面为什么是另一个思想' },
    falsification: { type: 'string', description: '什么结果会证伪它' },
    prior_belief: { type: 'string', description: '你认为它有效/无效的先验,以及理由' },
  },
  required: ['family_id', 'name', 'angle', 'hypothesis', 'rule', 'new_indicator', 'grid',
             'holding', 'distinct_from_uptrend_pullback', 'falsification', 'prior_belief'],
}

phase('Propose')

const proposals = (await parallel(ANGLES.map((a) => () => agent(
  [CTX, '', RULES, '', '=== 你被指派的角度 ===', a.ask,
    '', '只提这一个角度的**一个**策略族。要求:具体到可以照着实现;阈值给数值;',
    '如果你觉得这个角度上最好的想法仍然大概率无效,也要提出来并说明理由。'].join('\n'),
  { label: 'propose:' + a.key, phase: 'Propose', schema: SPEC },
)))).filter(Boolean)

log('收到 ' + proposals.length + ' 个策略族提案,进入评委打分')

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          family_id: { type: 'string' },
          mechanism: { type: 'integer', description: '1-5:机制是否可信(有文献/经济学理由,不是数据挖掘)' },
          distinctness: { type: 'integer', description: '1-5:与 uptrend-pullback 是否机制上不同' },
          implementable: { type: 'integer', description: '1-5:能否只用手上的列(+≤2 新指标)精确实现' },
          falsifiable: { type: 'integer', description: '1-5:证伪判据是否清晰、不会被样本量含糊掉' },
          risks: { type: 'string', description: '最大的失败模式 / 前视风险 / 与现有族重叠的具体风险' },
        },
        required: ['family_id', 'mechanism', 'distinctness', 'implementable', 'falsifiable', 'risks'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['scores', 'notes'],
}

const LENSES = [
  { key: 'mechanism', ask: '视角一(机制):逐个审「这个价格形态为什么代表被低估」的论证。'
    + '凡是只有直觉没有机制的,mechanism ≤ 2。特别注意:很多「超卖反转」的说法在美股大盘股上被'
    + '反复证伪,你要指出哪些是已被证伪的老路。' },
  { key: 'overlap', ask: '视角二(重叠度):逐个判断它是否真的与 uptrend-pullback 不同。'
    + '判据:它的信号集会不会大概率落在「上升趋势中的回调」这个子集里?会不会只是换了个阈值?'
    + '给具体的重叠机制,而不是感觉。' },
  { key: 'implementable', ask: '视角三(可实施+前视):逐个检查规则能否只用手上的列精确实现;'
    + '新指标是否因果(只用 ≤T 数据);有没有隐藏的前视(例如用到当日的 high/low 去决定当日开盘入场、'
    + '用全样本分位做阈值、用未来才知道的池子)。' },
]

const judged = (await parallel(LENSES.map((l) => () => agent(
  [CTX, '', '=== 你是评委 ===', l.ask, '',
    '下面是 ' + proposals.length + ' 个提案(JSON):',
    JSON.stringify(proposals.map((p) => ({ ...p, rule: p.rule.slice(0, 700), grid: p.grid.slice(0, 300) })), null, 1),
    '', '对**每一个** family_id 给分(1-5)并写清最大的失败模式。'].join('\n'),
  { label: 'judge:' + l.key, phase: 'Judge', schema: JUDGE_SCHEMA },
)))).filter(Boolean)

const tally = {}
for (const j of judged) {
  for (const s of j.scores || []) {
    const t = tally[s.family_id] || (tally[s.family_id] = { total: 0, n: 0, notes: [] })
    t.total += s.mechanism + s.distinctness + s.implementable + s.falsifiable
    t.n += 1
    t.notes.push(s.risks)
  }
}
const ranking = Object.entries(tally)
  .map(([id, t]) => ({ id, avg: t.total / t.n, risks: t.notes }))
  .sort((a, b) => b.avg - a.avg)
log('评委完成,平均分排序:' + ranking.map((r) => r.id + '=' + r.avg.toFixed(1)).join(' '))

phase('Synthesize')

const final = await agent(
  [CTX, '', '=== 你是最后合成的角色 ===', '',
    '下面是 10 个提案的完整规格与三位评委的打分排序。',
    '', '【提案】', JSON.stringify(proposals, null, 1),
    '', '【评委排序】', JSON.stringify(ranking, null, 1),
    '', '【你的任务】',
    '选 **6~8 个**互补的策略族(不要 10 个都要:宁可少而不同,不要多而重复),输出最终规格。',
    '要求:',
    '1. 每个族的规则**精确到可以照着写代码**,用给定列名,阈值给数值。',
    '2. 重新检查一遍**互不重复**:两族如果预期信号集高度重叠,合并或砍掉一个。',
    '3. 保留至少 2 个**评委打分不高但机制上最有意思**的(探索性),以及 4~6 个高分的。',
    '4. 给每个族一个**预注册网格**(我会照着跑 WFV,参数取值由你定,但必须写死)。',
    '5. 明确哪些族需要新增指标(≤2 个/族,精确因果公式)。',
    '6. 排一个**实现顺序**(先实现哪些,为什么)。',
    '7. 最后写清:如果 6~8 个族**全都**打不过 uptrend-pullback 的当前最佳(+14.19% 池化 /',
    '   +8.15% 连续净值),那说明什么(这是很可能的结果,要说清它的含义)。',
  ].join('\n'),
  { label: 'synthesize:final-specs', phase: 'Synthesize', schema: {
    type: 'object',
    properties: {
      families: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            family_id: { type: 'string' },
            name: { type: 'string' },
            hypothesis: { type: 'string' },
            rule: { type: 'string', description: '精确到可实现的规则(用给定列名 + 数值阈值)' },
            new_indicator: { type: 'string' },
            grid: { type: 'string', description: '预注册网格:参数名 = [取值...]' },
            holding: { type: 'string' },
            distinctness: { type: 'string' },
            falsification: { type: 'string' },
            exploratory: { type: 'boolean' },
          },
          required: ['family_id', 'name', 'hypothesis', 'rule', 'new_indicator', 'grid',
                     'holding', 'distinctness', 'falsification', 'exploratory'],
        },
      },
      implement_order: { type: 'string' },
      if_all_fail: { type: 'string', description: '如果所有族都打不过当前最佳,说明什么' },
    },
    required: ['families', 'implement_order', 'if_all_fail'],
  } },
)

return { proposals, ranking, final }
